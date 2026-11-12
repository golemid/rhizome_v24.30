#include "core/headless_run.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/algorithm_io.hpp"
#include "core/app_state.hpp"
#include "core/engine_resources.hpp"
#include "core/growth_controller.hpp"
#include "engine/kelp/kelp_session.hpp"
#include "engine/io/obj_exporter.hpp"
#include "engine/io/parsers.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/narrow_band_sdf.hpp"
#include "engine/sdf/root_field.hpp"

namespace rhizome {
namespace {

using SteadyClock = std::chrono::steady_clock;

double ms_since(SteadyClock::time_point t0) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - t0).count();
}

bool is_volumetric_mode(int mode) {
    return mode == 6; // Kelp Marine FSPM — the only engine in this build
}

// ---------------------------------------------------------------------------
// Developmental parameter application (the Kelp engine).
// GrowthAlgorithmProfile does not carry these fields — that is the documented
// cartridge gap — so the CLI exposes them by name. The mapping is explicit on
// purpose: no reflection, no silent fallback, and an unknown key is a hard
// usage error rather than a no-op. Range clamping is deliberately NOT
// duplicated here: build_colonization_config_from_state already applies the
// same clamps the GUI does, so headless and interactive runs agree.
// ---------------------------------------------------------------------------
bool apply_dev_override(GrowthState& g, const std::string& key, double value, std::string& err) {
    const float f = static_cast<float>(value);
    const int i = static_cast<int>(std::lround(value));

    if (key == "dev_gsa_framework")              g.dev_gsa_framework = f;
    else if (key == "dev_gsa_lateral")           g.dev_gsa_lateral = f;
    else if (key == "dev_grav_gain")             g.dev_grav_gain = f;
    else if (key == "dev_noise")                 g.dev_noise = f;
    else if (key == "dev_branch_angle")          g.dev_branch_angle = f;
    else if (key == "dev_lateral_freq")          g.dev_lateral_freq = f;
    else if (key == "dev_vigor")                 g.dev_vigor = f;
    else if (key == "dev_murray_gamma")          g.dev_murray_gamma = f;
    else if (key == "dev_taper_shape")           g.dev_taper_shape = i;
    else if (key == "dev_min_print_radius")      g.dev_min_print_radius = f;
    else if (key == "dev_section_budget")        g.dev_section_budget = i;
    else if (key == "dev_lateral_caliber_ratio") g.dev_lateral_caliber_ratio = f;
    else if (key == "dev_order_end_ratio")       g.dev_order_end_ratio = f;
    else if (key == "dev_surface_step_mult")     g.dev_surface_step_mult = f;
    else if (key == "dev_vessel_ref_mm")         g.dev_vessel_ref_mm = f;
    else {
        err = "unknown developmental parameter '" + key + "'";
        return false;
    }
    return true;
}

// Splits "a=1,b=2" into pairs. Empty segments are ignored.
bool parse_kv_list(const std::string& text, std::vector<std::pair<std::string, double>>& out, std::string& err) {
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        const auto eq = item.find('=');
        if (eq == std::string::npos || eq == 0) {
            err = "expected name=value in '" + item + "'";
            return false;
        }
        const std::string key = item.substr(0, eq);
        try {
            const double v = std::stod(item.substr(eq + 1));
            out.emplace_back(key, v);
        } catch (const std::exception&) {
            err = "not a number in '" + item + "'";
            return false;
        }
    }
    return true;
}

std::string newest_structure_dump(int mode) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const std::string prefix = "structure_dump_mode" + std::to_string(mode) + "_";
    std::string best;
    for (const auto& entry : fs::directory_iterator("logs", ec)) {
        if (ec) break;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0 && name > best) best = name;
    }
    return best.empty() ? std::string() : ("logs/" + best);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// One growth pass. The engine is not bit-reproducible run-to-run (CUDA
// compaction ordering), so a measurement must be a distribution, not a point.
struct RunResult {
    int iterations = 0;
    size_t sections = 0;
    size_t verts = 0;
    size_t tris = 0;
    bool manifold = true;
    bool watertight = true;
    size_t boundary_edges = 0;
    size_t non_manifold_edges = 0;
    glm::vec3 bmin{0.0f};
    glm::vec3 bmax{0.0f};
    double growth_ms = 0.0;
    double extract_ms = 0.0;
};

struct Spread {
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

Spread spread_of(const std::vector<double>& v) {
    Spread s;
    if (v.empty()) return s;
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (const double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    return s;
}

void emit_spread(std::ostream& os, const char* key, const Spread& s, bool last) {
    os << "    \"" << key << "\": { \"mean\": " << s.mean
       << ", \"min\": " << s.min << ", \"max\": " << s.max << " }"
       << (last ? "\n" : ",\n");
}

bool write_stats_json(const HeadlessRunOptions& opt,
                      const GrowthAlgorithmProfile& profile,
                      const GrowthState& growth,
                      size_t seed_count,
                      const std::vector<RunResult>& runs,
                      const std::string& dump_path,
                      double sdf_ms,
                      double export_ms,
                      double total_ms,
                      bool manifold_enforced) {
    std::ofstream js(opt.stats_path);
    if (!js) return false;

    std::vector<double> v_sections, v_verts, v_tris, v_iter, v_growth, v_bbox_z;
    for (const auto& r : runs) {
        v_sections.push_back(static_cast<double>(r.sections));
        v_verts.push_back(static_cast<double>(r.verts));
        v_tris.push_back(static_cast<double>(r.tris));
        v_iter.push_back(static_cast<double>(r.iterations));
        v_growth.push_back(r.growth_ms);
        v_bbox_z.push_back(static_cast<double>(r.bmax.z - r.bmin.z));
    }
    const Spread sp_sec = spread_of(v_sections);
    const Spread sp_ver = spread_of(v_verts);
    const Spread sp_tri = spread_of(v_tris);
    const Spread sp_itr = spread_of(v_iter);
    const Spread sp_gro = spread_of(v_growth);
    const Spread sp_bz  = spread_of(v_bbox_z);

    const RunResult& last = runs.back();

    js << std::fixed << std::setprecision(6);
    js << "{\n";
    js << "  \"rhizome_headless_run\": 1,\n";
    js << "  \"algorithm_file\": \"" << json_escape(opt.algorithm_path) << "\",\n";
    js << "  \"algorithm_name\": \"" << json_escape(profile.name) << "\",\n";
    js << "  \"mode\": " << growth.algorithm_mode << ",\n";
    js << "  \"seed\": " << opt.seed << ",\n";
    js << "  \"detail_mm\": " << opt.detail_mm << ",\n";
    js << "  \"vessel\": \""
       << json_escape(opt.vessel_path.empty() ? std::string("procedural:filleted_box_40mm")
                                              : opt.vessel_path) << "\",\n";
    js << "  \"repeat\": " << runs.size() << ",\n";
    js << "  \"seed_points\": " << seed_count << ",\n";
    js << "  \"determinism_note\": \"growth is not bit-reproducible run-to-run; compare the aggregate spread, not one run\",\n";

    // Aggregate over repeats — the number a parameter study should be read from.
    js << "  \"aggregate\": {\n";
    emit_spread(js, "sections", sp_sec, false);
    emit_spread(js, "vertices", sp_ver, false);
    emit_spread(js, "triangles", sp_tri, false);
    emit_spread(js, "iterations", sp_itr, false);
    emit_spread(js, "bbox_z_mm", sp_bz, false);
    emit_spread(js, "growth_ms", sp_gro, true);
    js << "  },\n";

    // Per-run detail, so a wide spread is visible rather than hidden.
    js << "  \"runs\": [\n";
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        js << "    { \"index\": " << i
           << ", \"iterations\": " << r.iterations
           << ", \"sections\": " << r.sections
           << ", \"vertices\": " << r.verts
           << ", \"triangles\": " << r.tris
           << ", \"manifold\": " << (r.manifold ? "true" : "false")
           << ", \"watertight\": " << (r.watertight ? "true" : "false")
           << ", \"growth_ms\": " << r.growth_ms
           << ", \"extract_ms\": " << r.extract_ms << " }"
           << (i + 1 == runs.size() ? "\n" : ",\n");
    }
    js << "  ],\n";

    // The exported mesh corresponds to the LAST run only.
    js << "  \"exported_mesh\": {\n";
    js << "    \"vertices\": " << last.verts << ",\n";
    js << "    \"triangles\": " << last.tris << ",\n";
    js << "    \"manifold\": " << (last.manifold ? "true" : "false") << ",\n";
    js << "    \"watertight\": " << (last.watertight ? "true" : "false") << ",\n";
    js << "    \"boundary_edges\": " << last.boundary_edges << ",\n";
    js << "    \"non_manifold_edges\": " << last.non_manifold_edges << ",\n";
    js << "    \"manifold_enforced_on_export\": " << (manifold_enforced ? "true" : "false") << "\n";
    js << "  },\n";
    js << "  \"bbox_mm\": {\n";
    js << "    \"min\": [" << last.bmin.x << ", " << last.bmin.y << ", " << last.bmin.z << "],\n";
    js << "    \"max\": [" << last.bmax.x << ", " << last.bmax.y << ", " << last.bmax.z << "],\n";
    js << "    \"size\": [" << (last.bmax.x - last.bmin.x) << ", "
       << (last.bmax.y - last.bmin.y) << ", " << (last.bmax.z - last.bmin.z) << "]\n";
    js << "  },\n";
    js << "  \"dev\": {\n";
    js << "    \"gsa_framework\": " << growth.dev_gsa_framework << ",\n";
    js << "    \"gsa_lateral\": " << growth.dev_gsa_lateral << ",\n";
    js << "    \"grav_gain\": " << growth.dev_grav_gain << ",\n";
    js << "    \"noise\": " << growth.dev_noise << ",\n";
    js << "    \"branch_angle\": " << growth.dev_branch_angle << ",\n";
    js << "    \"lateral_freq\": " << growth.dev_lateral_freq << ",\n";
    js << "    \"vigor\": " << growth.dev_vigor << ",\n";
    js << "    \"murray_gamma\": " << growth.dev_murray_gamma << ",\n";
    js << "    \"taper_shape\": " << growth.dev_taper_shape << ",\n";
    js << "    \"min_print_radius\": " << growth.dev_min_print_radius << ",\n";
    js << "    \"section_budget\": " << growth.dev_section_budget << ",\n";
    js << "    \"lateral_caliber_ratio\": " << growth.dev_lateral_caliber_ratio << ",\n";
    js << "    \"order_end_ratio\": " << growth.dev_order_end_ratio << ",\n";
    js << "    \"surface_step_mult\": " << growth.dev_surface_step_mult << ",\n";
    js << "    \"vessel_ref_mm\": " << growth.dev_vessel_ref_mm << "\n";
    js << "  },\n";
    js << "  \"structure_dump\": \"" << json_escape(dump_path) << "\",\n";
    js << "  \"timings_ms\": {\n";
    js << "    \"sdf_build\": " << sdf_ms << ",\n";
    js << "    \"growth_mean\": " << sp_gro.mean << ",\n";
    js << "    \"export\": " << export_ms << ",\n";
    js << "    \"total\": " << total_ms << "\n";
    js << "  }\n";
    js << "}\n";
    return true;
}

int execute(const HeadlessRunOptions& opt) {
    if (opt.algorithm_path.empty()) {
        std::cerr << "[--run] ERROR: --run requires an algorithm file (.json or .cu)\n";
        return 2;
    }

    const auto t_total = SteadyClock::now();

    // --- 1. algorithm profile (the "cartridge") ---------------------------
    GrowthAlgorithmProfile profile;
    if (!import_algorithm_from_file(opt.algorithm_path, profile)) {
        std::cerr << "[--run] ERROR: could not import algorithm file: " << opt.algorithm_path << "\n";
        return 3;
    }

    AppState st;
    EngineResources res;
    GrowthControllerContext ctx{st, res};

    st.growth.growth_algorithms.clear();
    st.growth.growth_algorithms.push_back(profile);
    apply_algorithm_profile_to_state(st, st.growth.growth_algorithms, 0);

    if (opt.mode_override >= 0)          st.growth.algorithm_mode = opt.mode_override;
    if (opt.steps_override > 0)          st.growth.max_iterations = opt.steps_override;
    if (opt.taper_curve_override > 0.0f) st.growth.taper_curve = opt.taper_curve_override;
    st.growth.detail_level = opt.detail_mm;

    for (const auto& kv : opt.dev_overrides) {
        std::string err;
        if (!apply_dev_override(st.growth, kv.first, kv.second, err)) {
            std::cerr << "[--run] ERROR: " << err << "\n";
            return 2;
        }
    }

    const int mode = st.growth.algorithm_mode;
    if (!is_volumetric_mode(mode)) {
        std::cerr << "[--run] ERROR: mode " << mode
                  << " is not available; --run supports only the Kelp Marine "
                     "FSPM engine (mode 6).\n";
        return 2;
    }

    const int repeats = std::max(1, opt.repeat);
    std::cout << "[--run] algorithm '" << profile.name << "', mode " << mode
              << ", detail " << opt.detail_mm << " mm, seed " << opt.seed
              << ", repeats " << repeats << "\n";

    // --- 2. vessel specimen ----------------------------------------------
    TriangleMesh vessel;
    if (!opt.vessel_path.empty()) {
        const std::string ext = std::filesystem::path(opt.vessel_path).extension().string();
        std::string parse_err;
        bool ok = false;
        if (ext == ".obj" || ext == ".OBJ")      ok = ObjParser::parse(opt.vessel_path, vessel, parse_err);
        else if (ext == ".stl" || ext == ".STL") ok = StlParser::parse(opt.vessel_path, vessel, parse_err);
        else {
            std::cerr << "[--run] ERROR: unsupported vessel extension '" << ext << "' (use .obj or .stl)\n";
            return 2;
        }
        if (!ok || vessel.vertex_count() == 0) {
            std::cerr << "[--run] ERROR: could not parse vessel mesh: " << parse_err << "\n";
            return 3;
        }
    } else {
        vessel = CudaRasterizer::generate_filleted_box_gpu(glm::vec3(20.0f), 2.0f, 48);
    }
    res.original_mesh = vessel;

    // --- 3. SDF (built once; repeats reuse it) ----------------------------
    const auto t_sdf = SteadyClock::now();
    NarrowBandSdf sdf_pipeline;
    SdfConfig sdf_cfg;
    sdf_cfg.voxel_size = opt.detail_mm;
    sdf_cfg.half_width_voxels = 4.0f;
    std::string sdf_log;
    auto vessel_grid = sdf_pipeline.build_from_mesh(vessel, sdf_cfg, sdf_log);
    const double sdf_ms = ms_since(t_sdf);
    if (!vessel_grid || !vessel_grid->is_valid()) {
        std::cerr << "[--run] ERROR: SDF generation failed: " << sdf_log << "\n";
        return 3;
    }
    res.current_sdf_grid = vessel_grid;

    // --- 4. seeds + kernel config (same mapping the GUI uses) -------------
    ensure_seed_points_controller(ctx);

    // Vessel-relative reference for the developmental length caps — the GUI
    // measures this at Start; headless must do the same or kelp strand lengths
    // fall back to the static table and the result differs.
    {
        const auto& g = *vessel_grid;
        const float ex = static_cast<float>(g.dims.x) * g.voxel_size;
        const float ey = static_cast<float>(g.dims.y) * g.voxel_size;
        const float ez = static_cast<float>(g.dims.z) * g.voxel_size;
        st.growth.dev_vessel_ref_mm = std::max(ex, std::max(ey, ez));
    }

    ColonizationConfig c_cfg =
        build_colonization_config_from_state(st, st.growth.growth_algorithms, st.seed.frozen);
    c_cfg.seed = opt.seed;

    const int batch_steps = c_cfg.max_iterations;

    // --- 5-6. growth + rasterize + extract, repeated ----------------------
    std::vector<RunResult> runs;
    runs.reserve(static_cast<size_t>(repeats));
    TriangleMesh root_mesh;
    float bbox_ms = 0.0f;

    for (int r = 0; r < repeats; ++r) {
        // Fresh session per repeat: a colonizer accumulates state, so reusing
        // one would make run N depend on run N-1.
        res.live_colonizer = SpaceColonization{};

        RunResult rr;
        const auto t_growth = SteadyClock::now();
        res.live_colonizer.initialize_volumetric_session(vessel_grid, st.seed.points, c_cfg);
        res.live_colonizer.step_volumetric_session(vessel_grid, c_cfg, opt.detail_mm,
                                                   st.growth.taper_curve, batch_steps);
        rr.growth_ms = ms_since(t_growth);

        const auto t_extract = SteadyClock::now();
        auto root_grid = res.live_colonizer.rasterize_volumetric_session(opt.detail_mm);
        auto confined_grid = RootImplicitField::intersect_with_vessel(root_grid, vessel_grid);
        root_mesh = NarrowBandSdf::extract_mesh(confined_grid, 0.0f, 0.0f);
        rr.extract_ms = ms_since(t_extract);

        rr.iterations = res.live_colonizer.get_current_iteration();
        rr.sections = res.live_colonizer.get_volumetric_sections().size();
        rr.verts = root_mesh.vertex_count();
        rr.tris = root_mesh.triangle_count();

        const ManifoldCheckResult mc = ObjExporter::verify_manifold(root_mesh);
        rr.manifold = mc.is_manifold;
        rr.watertight = mc.is_watertight;
        rr.boundary_edges = mc.boundary_edges;
        rr.non_manifold_edges = mc.non_manifold_edges;
        CudaRasterizer::compute_bounding_box_gpu(root_mesh, rr.bmin, rr.bmax, bbox_ms);

        std::cout << "[--run] run " << (r + 1) << "/" << repeats
                  << ": sections " << rr.sections << ", verts " << rr.verts
                  << ", growth " << rr.growth_ms << " ms\n";

        if (rr.verts == 0) {
            std::cerr << "[--run] ERROR: growth produced an empty mesh on run " << (r + 1) << "\n";
            return 3;
        }
        runs.push_back(rr);
    }

    // --- 7. export the last run -------------------------------------------
    const auto t_export = SteadyClock::now();
    std::string export_err;
    bool manifold_enforced = true;
    bool exported = ObjExporter::export_obj(opt.out_mesh_path, root_mesh, export_err, true);
    if (!exported) {
        // A non-manifold result is still worth inspecting, so retry without the
        // gate and record that the mesh failed it.
        manifold_enforced = false;
        exported = ObjExporter::export_obj(opt.out_mesh_path, root_mesh, export_err, false);
    }
    const double export_ms = ms_since(t_export);
    if (!exported) {
        std::cerr << "[--run] ERROR: OBJ export failed: " << export_err << "\n";
        return 3;
    }

    // --- 8. structure dump (reuses the GUI's own writer) ------------------
    write_structure_dump_controller(ctx, c_cfg);
    const std::string dump_path = newest_structure_dump(mode);

    // --- 9. stats ---------------------------------------------------------
    const double total_ms = ms_since(t_total);
    if (!write_stats_json(opt, profile, st.growth, st.seed.points.size(), runs, dump_path,
                          sdf_ms, export_ms, total_ms, manifold_enforced)) {
        std::cerr << "[--run] ERROR: could not write stats file: " << opt.stats_path << "\n";
        return 3;
    }

    const RunResult& last = runs.back();
    std::cout << "[--run] mode " << mode << " | last run: iterations " << last.iterations
              << ", sections " << last.sections << ", verts " << last.verts
              << ", manifold " << (last.manifold ? "YES" : "NO") << "\n";

    if (runs.size() > 1) {
        std::vector<double> sec, ver;
        for (const auto& r : runs) { sec.push_back(static_cast<double>(r.sections)); ver.push_back(static_cast<double>(r.verts)); }
        const Spread s_sec = spread_of(sec);
        const Spread s_ver = spread_of(ver);
        std::cout << "[--run] sections " << s_sec.min << ".." << s_sec.max
                  << " (mean " << s_sec.mean << "), verts " << s_ver.min << ".." << s_ver.max
                  << " (mean " << s_ver.mean << ") over " << runs.size() << " runs\n";
    }

    std::cout << "[--run] wrote " << opt.out_mesh_path << " and " << opt.stats_path;
    if (!dump_path.empty()) std::cout << " (structure dump: " << dump_path << ")";
    std::cout << "\n";
    std::cout << "[--run] timings: sdf " << sdf_ms << " ms, export " << export_ms
              << " ms, total " << total_ms << " ms\n";

    return 0;
}

} // namespace

int run_headless_cli(int argc, char** argv, int start) {
    HeadlessRunOptions opt;

    for (int i = start + 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };

        try {
            if (arg == "--run") {
                continue; // the flag itself; the file may follow positionally
            } else if (arg == "--mesh") {
                opt.vessel_path = next_value("--mesh");
            } else if (arg == "--out") {
                opt.out_mesh_path = next_value("--out");
            } else if (arg == "--stats") {
                opt.stats_path = next_value("--stats");
            } else if (arg == "--seed") {
                opt.seed = std::stoi(next_value("--seed"));
            } else if (arg == "--detail") {
                opt.detail_mm = std::stof(next_value("--detail"));
            } else if (arg == "--steps") {
                opt.steps_override = std::stoi(next_value("--steps"));
            } else if (arg == "--mode") {
                opt.mode_override = std::stoi(next_value("--mode"));
            } else if (arg == "--taper") {
                opt.taper_curve_override = std::stof(next_value("--taper"));
            } else if (arg == "--repeat") {
                opt.repeat = std::stoi(next_value("--repeat"));
            } else if (arg == "--dev") {
                std::string err;
                if (!parse_kv_list(next_value("--dev"), opt.dev_overrides, err)) {
                    std::cerr << "[--run] ERROR: " << err << "\n";
                    return 2;
                }
            } else if (!arg.empty() && arg[0] == '-') {
                std::cerr << "[--run] ERROR: unknown option '" << arg << "'\n";
                return 2;
            } else if (opt.algorithm_path.empty()) {
                opt.algorithm_path = arg; // first positional = algorithm file
            } else {
                std::cerr << "[--run] ERROR: unexpected extra argument '" << arg << "'\n";
                return 2;
            }
        } catch (const std::exception& e) {
            std::cerr << "[--run] ERROR: " << e.what() << "\n";
            return 2;
        }
    }

    if (opt.detail_mm <= 0.01f || opt.detail_mm > 2.0f) {
        std::cerr << "[--run] ERROR: --detail must be within (0.01, 2.0] mm\n";
        return 2;
    }
    if (opt.repeat < 1 || opt.repeat > 50) {
        std::cerr << "[--run] ERROR: --repeat must be within [1, 50]\n";
        return 2;
    }

    try {
        return execute(opt);
    } catch (const std::exception& e) {
        std::cerr << "[--run] ERROR: " << e.what() << "\n";
        return 3;
    } catch (...) {
        std::cerr << "[--run] ERROR: unknown failure\n";
        return 3;
    }
}

} // namespace rhizome

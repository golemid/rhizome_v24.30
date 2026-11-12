#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// =========================================================================
// GROWTH PRESETS SYSTEM (Presets storage in memory & imgui.ini)
// =========================================================================
struct GrowthPreset {
    std::string name;

    // Specimen detail
    float detail_level = 0.20f;

    // Seeds & Colonization
    int spore_count = 15000;
    int primary_roots = 4;
    float search_radius = 16.0f;
    float kill_radius = 1.2f;

    // Branching & Growth
    float growth_step = 1.60f;
    bool variable_step = false;
    float min_growth_step = 0.65f;
    float max_growth_step = 2.25f;
    float branching_rate = 0.45f;
    float steering_cone = 50.0f;
    int max_iterations = 120;

    // Vessel & Caliber
    float seed_radius = 1.40f;
    float trunk_caliber = 1.40f;
    float tendril_caliber = 0.35f;
    bool variable_taper = false;
    float taper_curve = 1.45f;
    float branch_smooth_k = 0.35f;
    float wall_clearance_scale = 0.90f;
    float wall_cut_smooth = 0.00f;

    // Seeding
    float seed_offset = 0.50f;

    // Growth Algorithm Mode & Physical Drivers
    int algorithm_mode = 6;
    // RETIRED / dead payload: the physical-driver fields below drive no kernel in the
    // single-engine build; retained only for INI/preset back-compat. Removal is a separate refactor.
    float root_roughness = 0.0f;
    float push_strength = 0.0f;
    float space_fill_drive = 0.60f;
    float backward_pull = 0.0f;
    float tensile_straightening = 0.0f;
    float murray_exponent = 3.0f;
    float murray_branch_angle = 37.5f;
    float surface_adherence = 0.0f;

    // Live growth simulation controls
    // v24.09 Track E: default OFF, matching GrowthState (app_state.hpp).
    // Presets/algorithm files saved without a LiveGrowth key now load
    // batch-baked; files that explicitly carry LiveGrowth=1 still apply it
    // (the importer overrides only when the key is present).
    bool live_growth = false;
    int live_steps_per_frame = 1;
    int live_fps = 8;

    // Developmental (Kelp Marine FSPM) parameters
    // v24.26: presets now capture the 14 developmental fields that drive the
    // single remaining engine. The dev_block_present flag gates restoration so
    // older preset files (which never saved these values) continue to load
    // safely — Apply leaves the developmental sliders untouched when the flag
    // is false, preserving v24.25 behavior for legacy presets.
    bool dev_block_present = false;
    float dev_gsa_framework = 55.0f;
    float dev_gsa_lateral = 68.0f;
    float dev_grav_gain = 0.15f;
    float dev_noise = 0.10f;
    float dev_branch_angle = 55.0f;
    float dev_lateral_freq = 1.0f;
    float dev_vigor = 1.0f;
    float dev_murray_gamma = 2.80f;
    int dev_taper_shape = 1;
    float dev_min_print_radius = 0.30f;
    int dev_section_budget = 30000;
    float dev_lateral_caliber_ratio = 1.00f;
    float dev_order_end_ratio = 1.00f;
    float dev_surface_step_mult = 1.12f;
};

void save_growth_presets_to_ini(const std::vector<GrowthPreset>& presets, const std::string& filepath = "imgui.ini");
void load_growth_presets_from_ini(std::vector<GrowthPreset>& presets, const std::string& filepath = "imgui.ini");

// =========================================================================
// VIEWPORT THEME PERSISTENCE (v24.01 Phase 0.4)
// =========================================================================
// [ViewportColors] and [LightingAndShading] are written by the Styles
// panel's "Copy Layout INI" exporter (panel_styles.cpp; GlassFrosting and
// SpecimenOpacity included since v24.01). This loader is the read side:
// paste the exported INI text into imgui.ini and the viewport theme plus
// the 3D lighting rig survive restarts. Applied only when the sections are
// present — fresh installs keep the compiled DisplayState defaults, and
// keys absent from a present section leave the current value untouched
// (same convention as load_growth_presets_from_ini).
namespace rhizome {
    struct DisplayState;  // core/app_state.hpp (co-located include cycle guard)
}
void load_viewport_style_from_ini(rhizome::DisplayState& display, const std::string& filepath = "imgui.ini");

// =========================================================================
// GROWTH ALGORITHMS SYSTEM (Export/Import in .json or .cu CUDA Source Format)
// =========================================================================
struct GrowthAlgorithmProfile {
    std::string name;
    int engine_mode = 6; // This build ships one engine: 6 = Kelp Marine (FSPM).
                         // All other engine ids were removed in earlier
                         // versions. The importer coerces any other value to 6
                         // (see apply_kv_to_algorithm).
    int primary_roots = 3; // Number of primary root arms spawned per seed [1..8]
    float root_roughness = 0.28f;
    float push_strength = 0.65f;
    float space_fill_drive = 0.90f;
    float backward_pull = 1.0f;
    float tensile_straightening = 0.0f;
    float murray_exponent = 3.0f;
    float murray_branch_angle = 37.5f;
    float surface_adherence = 0.0f; // [0..1] Attraction to and crawling along frozen root surfaces
    int spore_count = 25000;
    float search_radius = 16.0f;
    float kill_radius = 2.6f;
    float growth_step = 1.45f;
    bool variable_step = false;
    float min_growth_step = 0.65f;
    float max_growth_step = 2.25f;
    float branching_rate = 0.58f;
    float steering_cone = 35.0f;
    int max_iterations = 45;
    float seed_radius = 1.40f;
    float trunk_caliber = 1.00f;
    float tendril_caliber = 0.35f;
    bool variable_taper = false;
    float taper_curve = 1.45f;
    float seed_offset = 0.50f;

    // Internal CUDA GPU colonization kernel constants & growth rules
    float tip_inertia_weight = 1.25f;
    float spore_pull_weight = 0.75f;
    float roughness_amplitude = 0.65f;
    float feeler_nar_fwd = 0.92f;
    float feeler_nar_side = 0.38f;
    float feeler_med_fwd = 0.72f;
    float feeler_med_side = 0.69f;
    float feeler_wide_fwd = 0.42f;
    float feeler_wide_side = 0.90f;
    float feeler_wall_in = 0.55f;
    float feeler_wall_out = 0.45f;
    float clearance_score_weight = 2.50f;
    float align_bonus_weight = 0.65f;
    float lateral_fwd_weight = 0.52f;
    float lateral_side_weight = 0.85f;
    float lateral_radius_scale = 0.86f;
    int max_branch_generations = 4;
    float skin_gap_scale = 0.75f;
    float push_base_gain = 0.25f;
    float push_strength_gain = 0.55f;
    float max_step_disp_ratio = 0.28f;
    float tether_max_stretch = 1.25f;
    float tether_target_stretch = 1.15f;
    float wall_clearance_scale = 0.85f;
    float branch_smooth_k = 0.00f;
    float wall_cut_smooth = 0.00f;

    // ---------------------------------------------------------------------
    // Developmental block (modes 5/6). Optional: files written before this
    // block existed leave dev_block_present false, and the mode-based
    // default for dev_grav_gain then applies instead (see
    // apply_algorithm_profile_to_state). When the block IS present these
    // values win, so a saved algorithm reproduces its own run.
    // ---------------------------------------------------------------------
    bool  dev_block_present = false;
    float dev_gsa_framework = 55.0f;
    float dev_gsa_lateral = 68.0f;
    float dev_grav_gain = 0.00f;
    float dev_noise = 0.10f;
    float dev_branch_angle = 55.0f;
    float dev_lateral_freq = 1.0f;
    float dev_vigor = 1.0f;
    float dev_murray_gamma = 2.80f;
    int   dev_taper_shape = 1;
    float dev_min_print_radius = 0.30f;
    int   dev_section_budget = 30000;
    float dev_lateral_caliber_ratio = 1.00f;
    float dev_order_end_ratio = 1.00f;
    float dev_surface_step_mult = 1.12f;
};

using CustomAlgorithmProfile = GrowthAlgorithmProfile;

bool export_algorithm_to_json(const GrowthAlgorithmProfile& a, const std::string& filepath);
bool export_algorithm_to_cu(const GrowthAlgorithmProfile& a, const std::string& filepath);
bool export_algorithm_to_file(const GrowthAlgorithmProfile& a, const std::string& filepath);
void apply_kv_to_algorithm(const std::unordered_map<std::string, std::string>& kv, const std::string& fallback_name, GrowthAlgorithmProfile& out_algo);
bool import_algorithm_from_file(const std::string& filepath, GrowthAlgorithmProfile& out_algo);
void save_growth_algorithms_to_ini(const std::vector<GrowthAlgorithmProfile>& algos, int selected_idx, const std::string& filepath = "imgui.ini");
// v24.22: heal profiles whose engine_mode was corrupted by the historical
// 0..3 reload clamp. The factory profile is identified by its exact name,
// so a name match pins the engine mode (Kelp Marine FSPM -> 6). INIs
// written while the clamp bug was live can carry "Kelp (Marine FSPM)"
// with EngineMode=3; without this repair such an INI would load a retired
// engine under the Kelp name and defeat the factory default.
void repair_factory_profile_modes(std::vector<GrowthAlgorithmProfile>& algos);
// v24.22: factory_default_pending (optional) reports whether the INI carries
// no usable, marker-backed user choice: true when the file/section is
// missing, or when the [GrowthAlgorithms] section has no DefaultApplied key
// (i.e. an INI written before the one-time v24.22 factory default ran). The
// saver stamps DefaultApplied=2422 into the section it writes, so the flag
// is true exactly until the user's own first algorithm choice is saved.
void load_growth_algorithms_from_ini(std::vector<GrowthAlgorithmProfile>& algos, int& selected_idx, const std::string& filepath = "imgui.ini", bool* factory_default_pending = nullptr);

namespace rhizome {
    using ::GrowthPreset;
    using ::GrowthAlgorithmProfile;
    using ::CustomAlgorithmProfile;
    using ::save_growth_presets_to_ini;
    using ::load_growth_presets_from_ini;
    using ::load_viewport_style_from_ini;
    using ::export_algorithm_to_json;
    using ::export_algorithm_to_cu;
    using ::export_algorithm_to_file;
    using ::apply_kv_to_algorithm;
    using ::import_algorithm_from_file;
    using ::save_growth_algorithms_to_ini;
    using ::repair_factory_profile_modes;
    using ::load_growth_algorithms_from_ini;
}

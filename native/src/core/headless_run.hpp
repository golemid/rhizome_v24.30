#pragma once
// =========================================================================
// RHIZOME HEADLESS RUN — agent-facing batch entry point (unreleased)
// =========================================================================
// Purpose: let an external agent (or a CI job) drive the growth engine
// without a window, a GL context, or the ImGui panels.
//
//   rhizome_app.exe --run algo.json [--mesh mug.obj] [--dev k=v,...]
//                   [--seed 42] [--detail 0.20] [--steps 100]
//                   [--mode 6] [--out mesh.obj] [--stats stats.json]
//
// Design constraints (AGENTS.md):
//   * additive only — no existing CLI flag changes behaviour;
//   * GPU-only — every geometry stage goes through the CUDA engine, the
//     same calls the verification harness uses (no CPU fallback path);
//   * no GL — this path never constructs a window or touches a GL object.
//
// The run mirrors the GUI's own parameter mapping: the algorithm file is
// loaded into a GrowthAlgorithmProfile, applied with
// apply_algorithm_profile_to_state(), and the kernel config is derived with
// build_colonization_config_from_state() — the exact same functions the
// panels drive. Nothing here re-implements the mapping.
// =========================================================================

#include <string>
#include <utility>
#include <vector>

namespace rhizome {

struct HeadlessRunOptions {
    std::string algorithm_path;   // .json or .cu algorithm profile (required)
    std::string vessel_path;      // .obj or .stl specimen; empty => procedural box
    std::string out_mesh_path = "run_out.obj";
    std::string stats_path = "run_stats.json";
    int   seed = 42;              // overrides the config's built-in 42
    float detail_mm = 0.20f;      // SDF voxel size / detail level
    int   steps_override = -1;    // overrides max_iterations when > 0
    int   mode_override = -1;     // overrides engine_mode when 0..6
    int   repeat = 1;             // growth passes to average; engine is non-deterministic
    float taper_curve_override = -1.0f;

    // Developmental (modes 5/6) overrides. These parameters are NOT part of
    // GrowthAlgorithmProfile — that is the documented cartridge gap — so the
    // CLI takes them explicitly rather than silently ignoring them.
    std::vector<std::pair<std::string, double>> dev_overrides;
};

// Parses arguments beginning at argv[start] (which must be "--run") and
// executes one headless growth run.
// Returns 0 on success, 2 on usage error, 3 on runtime failure.
int run_headless_cli(int argc, char** argv, int start);

} // namespace rhizome

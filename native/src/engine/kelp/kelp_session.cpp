#include "engine/kelp/kelp_session.hpp"
#include "core/unified_logger.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdio>




void SpaceColonization::initialize_volumetric_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const ColonizationConfig& config
) {
    nodes_.clear();
    spores_.clear();
    vol_sections_.clear();
    vol_active_tips_.clear();
    vol_next_root_id_ = 1;
    current_iter_ = 0;
    bonus_iterations_ = 0;
    min_vessel_sdf_ = 0.0f;

    if (!vessel_grid || !vessel_grid->is_valid()) {
        return;
    }

    const float vs = vessel_grid->voxel_size;
    min_vessel_sdf_ = vessel_grid->get_min_sdf();
    glm::vec3 min_w = glm::vec3(vessel_grid->min_coord) * vs;
    glm::vec3 max_w = glm::vec3(vessel_grid->max_coord) * vs;

    // 1. Sample 3D environment nutrient pockets inside vessel volume strictly on GPU
    if (config.spore_count > 0) {
        const float spore_margin = (min_vessel_sdf_ < 0.0f) ? std::max(-0.15f, min_vessel_sdf_ * 0.45f) : -0.05f;
        const int target_spores = config.spore_count;
        spores_.reserve(target_spores);
        if (!rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 2027u, spore_margin, spores_)) {
            std::cerr << "[CUDA Error] GPU spore generation failed in sample_spores_gpu.\n";
        }
    }

    std::vector<glm::vec3> seeds = seed_positions;
    if (seeds.empty()) {
        seeds.push_back((min_w + max_w) * 0.5f);
    }

    // 2. Confine initial seeds to vessel interior and spawn primary root tips on GPU
    std::vector<rhizome::CudaVolSection> gpu_secs;
    std::vector<rhizome::CudaVolTip> gpu_tips;
    if (rhizome::CudaRasterizer::init_volumetric_seeds_gpu(
            *vessel_grid, seeds, config.base_radius, min_vessel_sdf_,
            config.seed + 2027u, gpu_secs, gpu_tips, vol_next_root_id_, config.wall_clearance_scale, config.primary_roots)) {
        if (!config.frozen_seeds.empty()) {
            bool spawned_dummy = false;
            rhizome::CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
                *vessel_grid, seeds, config.frozen_seeds, config.base_radius, min_vessel_sdf_,
                config.seed + 2027u, config.wall_clearance_scale,
                gpu_secs, gpu_tips, vol_next_root_id_, spawned_dummy, config.primary_roots
            );
        }
        vol_sections_.resize(gpu_secs.size());
        for (size_t s = 0; s < gpu_secs.size(); ++s) {
            if (gpu_secs[s].parent_sec < 0) {
                gpu_secs[s].radius = (config.seed_radius > 0.0f) ? config.seed_radius : config.base_radius;
            }
        }
        std::memcpy(vol_sections_.data(), gpu_secs.data(), gpu_secs.size() * sizeof(VolCrossSection));
        vol_active_tips_.resize(gpu_tips.size());
        std::memcpy(vol_active_tips_.data(), gpu_tips.data(), gpu_tips.size() * sizeof(VolRootTip));
        num_seminal_sections_ = vol_sections_.size();
    }
}

bool SpaceColonization::step_volumetric_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config,
    float voxel_size,
    float taper_curve,
    int steps_to_advance
) {
    if (!vessel_grid || !vessel_grid->is_valid() || vol_sections_.empty()) {
        return false;
    }

    // v24.12 Track F R1: section-budget guard (bake-size safety). Once the
    // cap is reached growth halts cleanly; the message lands once per call.
    if (static_cast<int>(vol_sections_.size()) >= config.dev_section_budget) {
        std::cerr << "[Dev] Section budget " << config.dev_section_budget
                  << " reached (" << vol_sections_.size() << " sections) - growth halted.\n";
        rhizome::RhizomeLog("REACTION", ("Developmental section budget " + std::to_string(config.dev_section_budget) + " reached - growth halted (" + std::to_string(vol_sections_.size()) + " sections)").c_str());
        return false;
    }

    // Laterals are real anatomy grown by the Kelp engine — never stripped.
    // num_seminal_sections_ stays a pure size snapshot.
    num_seminal_sections_ = vol_sections_.size();

    const float vs = vessel_grid->voxel_size;
    // The developmental program owns its timing: the exact iteration budget
    // (no space-fill scaling — that scaling belonged to the retired engines).
    const int base_steps = config.max_iterations;
    const int total_steps = base_steps + bonus_iterations_;

    if (current_iter_ >= total_steps) {
        return false;
    }

    // Radii carry the positional developmental caliber (Trunk Radius
    // anchored at each primary arm, tapering to Tip Radius at every
    // strand's live growing end), rebuilt by apply_dev_murray_caliber after
    // each step batch and at every slider sync, so both caliber sliders are
    // exact at every step of growth. Retired roots stay retired — the old
    // pre-step wake/respawn pass would regrow them.
    const float safe_taper = std::max(0.2f, taper_curve);
    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);

    (void)voxel_size;

    for (int s_step = 0; s_step < steps_to_advance; ++s_step) {
        bool any_awake_tip = false;
        rhizome::CudaRasterizer::check_any_awake_tips_gpu(gpu_tips_ref.data(), gpu_tips_ref.size(), any_awake_tip);
        if (!any_awake_tip || current_iter_ >= total_steps) {
            break;
        }

        std::mt19937 rng(config.seed + 2027u + static_cast<uint32_t>(current_iter_ * 131));

        // Execute the 3D Volumetric Root Colonization step directly on the NVIDIA CUDA GPU
        if (rhizome::CudaRasterizer::is_cuda_available()) {
            std::shuffle(vol_active_tips_.begin(), vol_active_tips_.end(), rng);

            rhizome::CudaVolGrowthParams gpu_params{};
            gpu_params.search_radius = config.search_radius;
            gpu_params.kill_radius = config.kill_radius;
            gpu_params.growth_step = config.growth_step;
            gpu_params.variable_step = config.variable_step;
            gpu_params.min_growth_step = config.min_growth_step;
            gpu_params.max_growth_step = config.max_growth_step;
            gpu_params.branching_rate = config.branching_rate;
            gpu_params.base_radius = config.base_radius;
            gpu_params.tip_radius = config.tip_radius;
            gpu_params.variable_taper = config.variable_taper;
            gpu_params.root_roughness = config.root_roughness;
            gpu_params.push_strength = config.push_strength;
            gpu_params.space_fill_drive = config.space_fill_drive;
            gpu_params.taper_curve = safe_taper;
            gpu_params.voxel_size = vs;
            gpu_params.min_vessel_sdf = min_vessel_sdf_;
            gpu_params.total_steps = total_steps;
            gpu_params.current_iter = current_iter_;
            gpu_params.seed = config.seed;
            gpu_params.tip_inertia_weight = config.tip_inertia_weight;
            gpu_params.spore_pull_weight = config.spore_pull_weight;
            gpu_params.roughness_amplitude = config.roughness_amplitude;
            gpu_params.feeler_nar_fwd = config.feeler_nar_fwd;
            gpu_params.feeler_nar_side = config.feeler_nar_side;
            gpu_params.feeler_med_fwd = config.feeler_med_fwd;
            gpu_params.feeler_med_side = config.feeler_med_side;
            gpu_params.feeler_wide_fwd = config.feeler_wide_fwd;
            gpu_params.feeler_wide_side = config.feeler_wide_side;
            gpu_params.feeler_wall_in = config.feeler_wall_in;
            gpu_params.feeler_wall_out = config.feeler_wall_out;
            gpu_params.clearance_score_weight = config.clearance_score_weight;
            gpu_params.align_bonus_weight = config.align_bonus_weight;
            gpu_params.lateral_fwd_weight = config.lateral_fwd_weight;
            gpu_params.lateral_side_weight = config.lateral_side_weight;
            gpu_params.lateral_radius_scale = config.lateral_radius_scale;
            gpu_params.max_branch_generations = config.max_branch_generations;
            gpu_params.skin_gap_scale = config.skin_gap_scale;
            gpu_params.push_base_gain = config.push_base_gain;
            gpu_params.push_strength_gain = config.push_strength_gain;
            gpu_params.max_step_disp_ratio = config.max_step_disp_ratio;
            gpu_params.tether_max_stretch = config.tether_max_stretch;
            gpu_params.tether_target_stretch = config.tether_target_stretch;
            gpu_params.wall_clearance_scale = config.wall_clearance_scale;
            gpu_params.steering_cone = config.steering_cone;
            gpu_params.backward_pull = config.backward_pull;
            gpu_params.tensile_straightening = config.tensile_straightening;
            gpu_params.algorithm_mode = config.algorithm_mode;
            gpu_params.murray_exponent = config.murray_exponent;
            gpu_params.murray_branch_angle = config.murray_branch_angle;
            gpu_params.surface_adherence = config.surface_adherence;
            // v24.12 Track F R1: developmental table onto the device params
            for (int o = 0; o < 4; ++o) {
                gpu_params.dev_gsa_deg[o] = config.dev_gsa_deg[o];
                gpu_params.dev_rate[o] = config.dev_rate[o];
                gpu_params.dev_spacing[o] = config.dev_spacing[o];
                gpu_params.dev_order_radius[o] = config.dev_order_radius[o];
                gpu_params.dev_tip_radius[o] = config.dev_tip_radius[o];
                gpu_params.dev_max_steps[o] = config.dev_max_steps[o];
            }
            gpu_params.dev_grav_gain = config.dev_grav_gain;
            gpu_params.dev_noise = config.dev_noise;
            gpu_params.dev_branch_angle = config.dev_branch_angle;
            gpu_params.dev_murray_gamma = config.dev_murray_gamma;
            gpu_params.dev_min_print_radius = config.dev_min_print_radius;
            gpu_params.dev_section_budget = config.dev_section_budget;
            gpu_params.dev_surface_step_mult = config.dev_surface_step_mult;

            if (rhizome::CudaRasterizer::step_volumetric_colonization_gpu(
                    *vessel_grid, gpu_params, spores_, gpu_secs_ref, gpu_tips_ref, vol_next_root_id_)) {
                current_iter_++;
                continue;
            } else {
                std::cerr << "[CUDA Error] GPU colonization step failed in step_volumetric_colonization_gpu.\n";
                break;
            }
        } else {
            std::cerr << "[CUDA Error] CUDA is not available for GPU colonization step.\n";
            break;
        }
    }

    // Final pass: rebuild the positional developmental caliber so live-view
    // radii already carry the slider-exact taper at every step of growth
    // (Trunk Radius locked at every primary arm, Tip Radius at every
    // strand's live leading tip — never tip-counted, never inflated).
    if (!apply_dev_murray_caliber(config)) {
        std::cerr << "[CUDA Error] Murray caliber pass failed in apply_dev_murray_caliber.\n";
    }

    bool any_awake_remaining = false;
    rhizome::CudaRasterizer::check_any_awake_tips_gpu(gpu_tips_ref.data(), gpu_tips_ref.size(), any_awake_remaining);

    // Snapshot the root count. There is no decoration-lateral pass in the
    // Kelp engine — laterals are real anatomy grown by the dev kernels.
    num_seminal_sections_ = vol_sections_.size();

    return (any_awake_remaining && current_iter_ < total_steps);
}

std::shared_ptr<rhizome::CudaVolumeGrid> SpaceColonization::rasterize_volumetric_session(
    float voxel_size,
    bool preserve_beaded,
    float branch_smooth_k
) const {
    auto volume = std::make_shared<rhizome::CudaVolumeGrid>();
    volume->voxel_size = voxel_size;
    if (vol_sections_.empty()) {
        return volume;
    }

    float cuda_ms = 0.0f;
    rhizome::CudaRasterizer::rasterize_volumetric_sections_gpu(
        reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
        vol_sections_.size(),
        voxel_size,
        *volume,
        cuda_ms,
        preserve_beaded,
        branch_smooth_k
    );

    return volume;
}


void SpaceColonization::sample_spores_only(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config
) {
    spores_.clear();
    if (!vessel_grid || !vessel_grid->is_valid()) return;

    const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
    const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
    const int target_spores = config.spore_count;
    spores_.reserve(target_spores);

    rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 2027u, spore_margin, spores_);
}

void SpaceColonization::prepare_resume_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const ColonizationConfig& config
) {
    if (!vessel_grid || !vessel_grid->is_valid()) return;

    const int base_steps = config.max_iterations;

    if (config.max_iterations > current_iter_) {
        // User increased the Max Steps slider: grow exactly to the new slider target!
        bonus_iterations_ = std::max(bonus_iterations_, config.max_iterations - base_steps);
    } else {
        // User clicked Start without changing the slider: grant a 25-step extension
        bonus_iterations_ = std::max(bonus_iterations_, (current_iter_ - base_steps) + 25);
    }

    // Replenish spores if depleted so continuing root tips have fresh nutrients to chase
    int min_needed_spores = std::max(250, config.spore_count / 10);
    if (static_cast<int>(spores_.size()) < min_needed_spores) {
        const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
        const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
        const int target_spores = std::max(min_needed_spores * 2, config.spore_count / 3);

        std::vector<glm::vec3> new_spores;
        if (rhizome::CudaRasterizer::sample_spores_gpu(*vessel_grid, target_spores, config.seed + 4099u + static_cast<uint32_t>(current_iter_ * 97), spore_margin, new_spores)) {
            spores_.insert(spores_.end(), new_spores.begin(), new_spores.end());
        } else {
            std::cerr << "[CUDA Error] GPU spore replenishment failed in sample_spores_gpu.\n";
        }
    }

    // Re-energize existing tips and awaken leaf tips on GPU if needed
    bool any_awake = false;
    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
    rhizome::CudaRasterizer::wake_or_respawn_volumetric_tips_gpu(
        gpu_secs_ref.data(), gpu_secs_ref.size(), gpu_tips_ref, true, any_awake
    );
}

void SpaceColonization::sync_seeds_session(
    std::shared_ptr<rhizome::CudaVolumeGrid> vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const ColonizationConfig& config
) {
    if (!vessel_grid || !vessel_grid->is_valid() || seed_positions.empty()) return;

    // If growth has not advanced any steps yet, cleanly re-initialize all seeds and tips on GPU
    if (current_iter_ == 0) {
        initialize_volumetric_session(vessel_grid, seed_positions, config);
        return;
    }

    bool spawned_new = false;
    if (vol_sections_.empty()) return;

    // Kelp laterals are anatomical sections and must never be trimmed.
    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
    rhizome::CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
        *vessel_grid, seed_positions, config.frozen_seeds, config.base_radius,
        vessel_grid->get_min_sdf(), config.seed + 2027u + static_cast<uint32_t>(current_iter_ * 31),
        config.wall_clearance_scale, gpu_secs_ref, gpu_tips_ref, vol_next_root_id_, spawned_new, config.primary_roots
    );

    if (spawned_new) {
        num_seminal_sections_ = vol_sections_.size();
    }

    if (spawned_new) {
        const int base_steps = config.max_iterations;
        if (current_iter_ + base_steps > base_steps + bonus_iterations_) {
            bonus_iterations_ = current_iter_;
        }

        // Replenish nutrient spores on GPU so the newly spawned seed has a rich nutrient cloud to grow into
        const float min_v_sdf = std::min(0.0f, vessel_grid->get_min_sdf());
        const float spore_margin = (min_v_sdf < 0.0f) ? std::max(-0.15f, min_v_sdf * 0.45f) : -0.05f;
        const int target_spores = std::max(1000, config.spore_count - static_cast<int>(spores_.size()));
        if (target_spores > 0) {
            std::vector<glm::vec3> new_spores;
            if (rhizome::CudaRasterizer::sample_spores_gpu(
                    *vessel_grid, target_spores, config.seed + 6151u + static_cast<uint32_t>(current_iter_ * 131),
                    spore_margin, new_spores)) {
                spores_.insert(spores_.end(), new_spores.begin(), new_spores.end());
            }
        }
    }
}


void SpaceColonization::delete_seed(int seed_index) {
    if (seed_index < 0) return;
    // v24.23: the classic node-pruning fallback was removed with the classic
    // engine. Seeds only exist inside a volumetric session now.
    if (vol_sections_.empty()) return;

    auto& gpu_secs_ref = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    auto& gpu_tips_ref = reinterpret_cast<std::vector<rhizome::CudaVolTip>&>(vol_active_tips_);
    rhizome::CudaRasterizer::prune_volumetric_sections_by_seed_gpu(gpu_secs_ref, gpu_tips_ref, seed_index);
    num_seminal_sections_ = vol_sections_.size();

    nodes_.resize(vol_sections_.size());
    if (!vol_sections_.empty()) {
        rhizome::CudaRasterizer::convert_vol_sections_to_nodes_gpu(
            reinterpret_cast<const rhizome::CudaVolSection*>(vol_sections_.data()),
            vol_sections_.size(),
            nodes_.data()
        );
    } else {
        nodes_.clear();
    }
}
void SpaceColonization::update_caliber(
    float trunk_caliber,
    float tip_radius,
    float taper_curve,
    bool variable_taper,
    uint32_t rng_seed,
    const ColonizationConfig* dev_cfg
) {
    if (vol_sections_.empty() && nodes_.empty()) return;

    // The Caliber sliders drive the positional developmental taper (Trunk
    // Radius anchors every primary arm; the retired depth-based taper would
    // fight the developmental scheme).
    (void)trunk_caliber;
    (void)tip_radius;
    (void)taper_curve;
    (void)variable_taper;
    (void)rng_seed;
    if (dev_cfg && !vol_sections_.empty()) {
        apply_dev_murray_caliber(*dev_cfg);
    }
}

// =========================================================================
// v24.18 — DEVELOPMENTAL POSITIONAL CALIBER, LIVE-SLIDER FIDELITY (replaces
// the v24.12 Murray tip-counting census) + SLA PRINT AUDIT
// =========================================================================

// v24.17: positional developmental caliber — the tip-accumulation Murray
// census is fully eliminated. The census inferred every section's radius
// from how many branch tips its subtree carried, so trunk arms started at
// Tip Radius (one tip) and inflated step after step as laterals spawned,
// while the Trunk Radius slider acted only as a distant ceiling (reaching
// it needed ~4^gamma tips) and the seed sphere towered over skinny trunk
// strings as a giant bulb.
//
// v24.18 refinement: BOTH caliber sliders are honored EXACTLY at every
// instant of the simulation — especially while a root is being generated.
// The taper is normalized over each strand's CURRENT grown length (not its
// final programmed length), so every frame — mid-growth included — shows
// the full slider profile end to end: Trunk Radius at the base, Tip Radius
// at the leading tip.
//
// The caliber is purely POSITIONAL, anchored to the user sliders:
//   1. Trunk base anchored from step 0 — the first section of every
//      primary arm (order 0) is locked exactly at the Trunk Radius slider
//      at every step of growth and can never inflate: a section's radius
//      is a pure function of its arc position, the current geometry and
//      the sliders. Nothing is counted, nothing accumulates, nothing is
//      carried between passes.
//   2. Smooth taper to Tip Radius at EVERY step — along each strand,
//      thickness falls smoothly from the strand anchor down to the strand
//      order's END radius over the strand's grown-so-far arc length, shaped
//      by the Murray Gamma falloff exponent. The leading tip of every
//      strand therefore sits exactly at its order's end radius WHILE the
//      root is being generated (not only when it completes its program),
//      and the profile stretches smoothly as the root elongates. A fixed
//      section only ever THICKENS toward its strand anchor as the cone
//      stretches — never thins or flickers — and a completed root's shape
//      is frozen by construction (its grown length stops changing).
//      Order 0 ends EXACTLY at the Tip Radius slider (the sliders'
//      promise); deeper orders end finer when Order End Thinning < 1.
//   3. No bulbous seed joint — primary arms start at Trunk Radius, so
//      with the Seed slider at the same value the arms leave the seed
//      sphere flush and the giant bulb disappears. Lateral strands anchor
//      at the parent's radius at the branch point (x the Lateral Caliber
//      Ratio lever, identity 1.00 keeps the joint flush), so every branch
//      joint is seamless at the default and a deliberate fine shoulder
//      appears when the ratio is dialed below 1.
//   4. Uniform clay snakes — Tip Radius >= Trunk Radius collapses every
//      root segment to an exact Trunk Radius cylinder everywhere.
//   5. v24.19 calibration levers (identity by default — the v24.18 look is
//      untouched until the user dials them):
//        - Lateral Caliber Ratio [0.30..1.00]: lateral birth caliber =
//          parent radius at the branch point x this. Real roots branch
//          thinner (~0.6-0.8); 1.00 keeps the flush v24.18 joint.
//        - Order End Thinning [0.40..1.00]: each order's taper end = its
//          parent order's end x this, clamped at the SLA print floor.
//          Order 0 always ends exactly at the Tip Radius slider.
//   6. v24.20 Taper Curve Shape: the profile FAMILY between a strand's
//      anchor and its order end — 0 Linear (even cone), 1 Natural power
//      (t^falloff, the default and the exact v24.18/v24.19 curve), 2
//      S-Curve (smoothstep), 3 Whip (early thinning). The endpoints never
//      move; the shape only redistributes caliber along the strand.
// Frozen sections (pinned bit 0) keep their locked caliber; seed bases
// ride the Seed Radius slider unless user-locked (pinned bit 1 set by
// freeze). Host-side O(N) double forward sweep (parents always precede
// children by the append discipline); fully deterministic.
bool SpaceColonization::apply_dev_murray_caliber(const ColonizationConfig& config) {
    if (vol_sections_.empty()) return false;

    auto& secs = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    const int n = static_cast<int>(secs.size());
    const float floor_r = std::max(0.05f, config.dev_min_print_radius);
    // User surface, verbatim: Trunk Radius anchors every primary arm,
    // Tip Radius terminates every primary strand (deeper orders thin further
    // when Order End Thinning is dialed below 1). No ceiling-cap semantics.
    const float trunk_r = std::max(floor_r, config.base_radius);
    const float tip_r = std::max(floor_r, config.tip_radius);
    // Murray Gamma stays live as the taper falloff shaper for the Natural
    // power shape (1 = linear cone, higher = thickness held longer before
    // easing into the tip). v24.20 adds the Taper Curve Shape dropdown: the
    // profile FAMILY between a strand's anchor and its order end. Endpoints
    // never move (anchor and order end come from the sliders); the shape
    // only redistributes caliber along the strand:
    //   0 Linear   — even cone, radius falls steadily with arc
    //   1 Natural  — t^falloff power curve (default, the v24.18/v24.19 math)
    //   2 S-Curve  — smoothstep, caliber lingers then lands softly
    //   3 Whip     — sqrt profile, thins early then runs fine to the tip
    const float falloff = std::min(4.0f, std::max(1.5f, config.dev_murray_gamma));
    const int shape = std::min(3, std::max(0, config.dev_taper_shape));
    auto taper_profile = [&](float t) -> float {
        switch (shape) {
            case 0: return t;                          // Linear: even cone
            case 2: return t * t * (3.0f - 2.0f * t);  // S-Curve: smoothstep
            case 3: return std::sqrt(t);               // Whip: early thinning
            default: return std::pow(t, falloff);      // Natural: power (default)
        }
    };

    // v24.19 calibration levers (identity defaults keep v24.18 behavior).
    const float lateral_ratio = std::min(1.00f, std::max(0.30f, config.dev_lateral_caliber_ratio));
    const float order_ratio = std::min(1.00f, std::max(0.40f, config.dev_order_end_ratio));
    // Per-order taper ends: order 0 ends exactly at the Tip Radius slider,
    // deeper orders end finer (parent order's end x the thinning ratio,
    // clamped at the SLA print floor).
    float end_by_order[4];
    end_by_order[0] = tip_r;
    for (int o = 1; o < 4; ++o) {
        end_by_order[o] = std::max(floor_r, end_by_order[o - 1] * order_ratio);
    }

    // Direct check: if Tip Radius equals or exceeds Trunk Radius (e.g. uniform clay snakes)
    const bool uniform_clay_snakes = (config.tip_radius >= config.base_radius - 1e-4f);
    if (uniform_clay_snakes) {
        // Exact uniform caliber across all branches: radius = config.base_radius
        for (int i = 0; i < n; ++i) {
            if (secs[i].parent_sec < 0) {
                const bool user_locked_seed = (secs[i].pinned & 2) != 0;
                if (!user_locked_seed) {
                    secs[i].radius = (config.seed_radius > 0.0f) ? config.seed_radius : config.base_radius;
                }
            } else {
                const bool frozen = (secs[i].pinned & 1) != 0;
                if (!frozen) {
                    secs[i].radius = config.base_radius;
                }
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------
    // Pass 1 — geometry census (no radius writes): arc length from the
    // seed, strand detection, and each strand's CURRENT grown extent.
    // Every primary arm and every lateral owns a fresh root_id (kernel
    // convention: continuations inherit tip.root_id, lateral children
    // atomicAdd a new one), so a strand start is exactly where root_id
    // differs from the parent. The grown extent is the largest arc offset
    // any current member carries — the live denominator that keeps both
    // sliders exact at every step of generation.
    // ---------------------------------------------------------------------
    std::vector<float> arc(static_cast<size_t>(n), 0.0f);        // mm from the seed base along the parent chain
    std::vector<int>   strand_anchor(static_cast<size_t>(n), -1); // index of the owning strand's anchor section
    std::vector<float> strand_origin(static_cast<size_t>(n), 0.0f); // arc position where the strand's taper starts
    std::vector<float> grown_extent(static_cast<size_t>(n), 0.0f); // per-anchor: strand's grown-so-far arc (mm)
    for (int i = 0; i < n; ++i) {
        const int p = secs[i].parent_sec;

        if (p < 0) {
            // Seed base sphere: not a strand member.
            arc[i] = 0.0f;
            strand_anchor[i] = -1;
            strand_origin[i] = 0.0f;
            continue;
        }

        if (p >= n) {
            // Dangling parent (defensive; pruning prevents this): treat as
            // an anchored strand start so the section keeps a sane caliber.
            arc[i] = 0.0f;
            strand_anchor[i] = i;
            strand_origin[i] = 0.0f;
            continue;
        }

        const glm::vec3 dvec = secs[i].pos - secs[p].pos;
        arc[i] = arc[p] + std::sqrt(glm::dot(dvec, dvec));

        float extent_candidate = 0.0f;
        if (secs[i].root_id != secs[p].root_id) {
            // Strand start: this section IS the anchor.
            strand_anchor[i] = i;
            strand_origin[i] = arc[i];
        } else {
            // Continuation along the strand (frozen or not): inherit the
            // lineage anchor and count toward the strand's grown extent.
            strand_anchor[i] = strand_anchor[p];
            strand_origin[i] = strand_origin[p];
            extent_candidate = arc[i] - strand_origin[i];
        }

        // Defensive guard (arms always own a fresh root_id, so a -1 lineage
        // cannot occur in practice): promote to a fresh strand anchor.
        if (strand_anchor[i] < 0) {
            strand_anchor[i] = i;
            strand_origin[i] = arc[i];
            extent_candidate = 0.0f;
        }

        float& ext = grown_extent[strand_anchor[i]];
        ext = std::max(ext, extent_candidate);
    }

    // ---------------------------------------------------------------------
    // Pass 2 — radius resolve (parents precede children, so every strand
    // anchor's radius is final before its continuations read it).
    // ---------------------------------------------------------------------
    std::vector<float> anchor_radius(static_cast<size_t>(n), 0.0f); // radius carried by each strand anchor
    for (int i = 0; i < n; ++i) {
        const int p = secs[i].parent_sec;
        const bool frozen = (secs[i].pinned & 1) != 0;

        if (p < 0) {
            // Seed base sphere: rides the Seed Radius slider (kept when user-locked).
            const bool user_locked_seed = (secs[i].pinned & 2) != 0;
            if (!user_locked_seed) {
                secs[i].radius = (config.seed_radius > 0.0f) ? config.seed_radius : trunk_r;
            }
            continue;
        }

        // Frozen sections keep their locked caliber completely intact, but
        // still publish it as the lineage anchor for any non-frozen
        // descendants branching off them.
        if (frozen) {
            anchor_radius[i] = secs[i].radius;
            continue;
        }

        const int order = std::min(std::max(secs[i].pinned >> 1, 0), 3);

        if (p >= n) {
            // Dangling parent (defensive): anchored strand start.
            const float anchor = (order == 0) ? trunk_r : end_by_order[order];
            secs[i].radius = std::max(floor_r, anchor);
            anchor_radius[i] = secs[i].radius;
            continue;
        }

        if (secs[i].root_id != secs[p].root_id || strand_anchor[i] == i) {
            // Strand anchor. Primary arms (order 0) lock hard at the Trunk
            // Radius slider from the very first step; lateral strands anchor
            // at the parent's radius at the branch point scaled by the
            // Lateral Caliber Ratio (1.00 = flush seamless joint; below 1.00
            // carves a deliberate fine shoulder at every branch point).
            const float anchor = (order == 0) ? trunk_r
                                              : std::max(floor_r, secs[p].radius * lateral_ratio);
            secs[i].radius = std::max(floor_r, anchor);
            anchor_radius[i] = secs[i].radius;
            continue;
        }

        // Continuation along the strand: positional taper over the strand's
        // CURRENT grown length. t = arc since the anchor over grown-so-far,
        // so the leading tip lands exactly on this order's end radius at
        // every step of generation and the profile stretches smoothly as the
        // root elongates (fixed sections only ever thicken toward the anchor
        // — no re-counting, no inflation past the anchor, no flicker, ever).
        // The selected shape (v24.20 Taper Curve Shape) redistributes caliber
        // between the anchor and the order's end; Natural's t^falloff holds
        // the anchor thickness through the trunk region and eases smoothly
        // into the order's end at the growing end (falloff = 1 would be a
        // plain linear cone). Strands born finer than their order's end
        // (thin parents, strong thinning) stay at their birth caliber end
        // to end.
        const int a = strand_anchor[i];
        const float base_r = (a >= 0) ? anchor_radius[a] : trunk_r;
        const float end_r = end_by_order[order];
        const float denom = std::max(grown_extent[a], 1e-6f);
        float t = (arc[i] - strand_origin[i]) / denom;
        t = std::min(1.0f, std::max(0.0f, t));
        const float r = (base_r <= end_r)
            ? base_r
            : end_r + (base_r - end_r) * (1.0f - taper_profile(t));
        secs[i].radius = std::max(floor_r, r);
    }

    return true;
}

void SpaceColonization::emit_print_audit(const ColonizationConfig& config) {
    if (vol_sections_.empty()) return;

    auto& secs = reinterpret_cast<std::vector<rhizome::CudaVolSection>&>(vol_sections_);
    const int n = static_cast<int>(secs.size());
    const float floor_r = std::max(0.05f, config.dev_min_print_radius);

    // Per-order census, resin volume, min radius, connectivity.
    int per_order[4] = {0, 0, 0, 0};
    float per_order_min[4] = {1e9f, 1e9f, 1e9f, 1e9f};
    double per_order_sum[4] = {0.0, 0.0, 0.0, 0.0};
    int seed_bases = 0;
    int parent_out_of_range = 0;
    float min_radius = 1e9f;
    double volume_mm3 = 0.0;
    for (int i = 0; i < n; ++i) {
        const int order = std::min(std::max(secs[i].pinned >> 1, 0), 3);
        ++per_order[order];
        per_order_min[order] = std::min(per_order_min[order], secs[i].radius);
        per_order_sum[order] += secs[i].radius;
        if (secs[i].parent_sec < 0) {
            ++seed_bases;
        } else {
            if (secs[i].parent_sec >= n) ++parent_out_of_range;
            // Segment volume: capsule from parent to this section, average radius
            const auto& p = secs[secs[i].parent_sec].pos;
            const auto& q = secs[i].pos;
            double seg_len = std::sqrt(static_cast<double>(
                (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y) + (q.z - p.z) * (q.z - p.z)));
            double r_avg = 0.5 * (static_cast<double>(secs[i].radius) + static_cast<double>(secs[secs[i].parent_sec].radius));
            volume_mm3 += 3.141592653589793 * r_avg * r_avg * seg_len;
        }
        if (secs[i].radius < min_radius) min_radius = secs[i].radius;
    }

    // Per-root slenderness (chain length / min radius of that root).
    // root_id groups are individual roots (commit assigns one per strand).
    // v24.19: RootStat also tracks the strand's leading tip (deepest
    // section), its birth caliber (first section seen — parents precede
    // children, so the first hit is the anchor) and its order, for the
    // caliber fidelity audit below.
    struct RootStat { double len; float min_r; int secs; int tip_idx = -1; int tip_depth = -1; float birth_r = -1.0f; int order = 0; };
    std::unordered_map<int, RootStat> roots;
    roots.reserve(static_cast<size_t>(n) / 8 + 16);
    for (int i = 0; i < n; ++i) {
        if (secs[i].parent_sec < 0) continue;
        const auto& p = secs[secs[i].parent_sec].pos;
        const auto& q = secs[i].pos;
        double seg_len = std::sqrt(static_cast<double>(
            (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y) + (q.z - p.z) * (q.z - p.z)));
        auto it = roots.find(secs[i].root_id);
        if (it == roots.end()) {
            roots.emplace(secs[i].root_id, RootStat{seg_len, secs[i].radius, 1, i, secs[i].depth, secs[i].radius, std::min(std::max(secs[i].pinned >> 1, 0), 3)});
        } else {
            it->second.len += seg_len;
            it->second.min_r = std::min(it->second.min_r, secs[i].radius);
            it->second.secs += 1;
            if (secs[i].depth > it->second.tip_depth) {
                it->second.tip_idx = i;
                it->second.tip_depth = secs[i].depth;
            }
        }
    }
    double worst_slen = 0.0; int worst_root = -1;
    int roots_over_60 = 0;
    for (const auto& kv : roots) {
        double slen = kv.second.len / std::max(0.05, static_cast<double>(kv.second.min_r));
        if (slen > 60.0) ++roots_over_60;
        if (slen > worst_slen) { worst_slen = slen; worst_root = kv.first; }
    }

    char line[512];
    std::snprintf(line, sizeof(line),
        "[Print Audit] sections=%d (order 0/1/2/3: %d/%d/%d/%d) | islands=%d seed base(s) %s | "
        "min radius=%.3f mm (floor %.3f) %s | est. resin volume=%.2f cm^3",
        n, per_order[0], per_order[1], per_order[2], per_order[3],
        seed_bases, (parent_out_of_range == 0 ? "connected" : "BROKEN"),
        min_radius, floor_r, (min_radius >= floor_r - 1e-4f ? "OK" : "BELOW"),
        volume_mm3 / 1000.0);
    rhizome::RhizomeLog("PRINT", line);
    std::snprintf(line, sizeof(line),
        "[Print Audit] slenderness: worst root #%d L/r=%.0f | roots with L/r > 60: %d "
        "(long thin spans - consider supports or a higher Min Print Radius)",
        worst_root, worst_slen, roots_over_60);
    rhizome::RhizomeLog("PRINT", line);

    // v24.19: caliber audit — per-order radius stats (min/mean) and leading-tip
    // fidelity vs the positional taper's per-order targets. Order 0 tips must
    // sit exactly at the Tip Radius slider; deeper orders at their thinned end
    // (parent order's end x Order End Thinning, floor-clamped). Strands born
    // finer than their target (thin parents) are measured against their birth
    // caliber, mirroring the pass's own clamp. Frozen tips keep their locked
    // caliber and are excluded from the fidelity count.
    {
        const float lateral_ratio = std::min(1.00f, std::max(0.30f, config.dev_lateral_caliber_ratio));
        const float order_ratio = std::min(1.00f, std::max(0.40f, config.dev_order_end_ratio));
        float end_by_order[4];
        end_by_order[0] = std::max(floor_r, config.tip_radius);
        for (int o = 1; o < 4; ++o) end_by_order[o] = std::max(floor_r, end_by_order[o - 1] * order_ratio);
        (void)lateral_ratio;  // birth caliber is read from the sections themselves

        static const char* k_taper_shape_names[4] = {"Linear", "Natural", "S-Curve", "Whip"};
        const int shape_sel = std::min(3, std::max(0, config.dev_taper_shape));
        int tips_checked[4] = {0, 0, 0, 0};
        float tip_dev_worst[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (const auto& kv : roots) {
            const RootStat& rs = kv.second;
            if (rs.tip_idx < 0 || rs.secs < 2 || (secs[rs.tip_idx].pinned & 1)) continue;
            const int o = std::min(std::max(rs.order, 0), 3);
            const float target = (rs.birth_r > 0.0f && rs.birth_r <= end_by_order[o]) ? rs.birth_r : end_by_order[o];
            ++tips_checked[o];
            tip_dev_worst[o] = std::max(tip_dev_worst[o], std::fabs(secs[rs.tip_idx].radius - target));
        }
        std::snprintf(line, sizeof(line),
            "[Caliber Audit] shape=%s | r min/mean by order 0/1/2/3: "
            "%.2f/%.2f %.2f/%.2f %.2f/%.2f %.2f/%.2f mm | tip fidelity: %d/%d/%d/%d strands, "
            "worst |tip-end| %.3f/%.3f/%.3f/%.3f mm (order 0 must stay 0.000)",
            k_taper_shape_names[shape_sel],
            per_order_min[0] < 1e8f ? per_order_min[0] : 0.0f, static_cast<float>(per_order[0] > 0 ? per_order_sum[0] / per_order[0] : 0.0),
            per_order_min[1] < 1e8f ? per_order_min[1] : 0.0f, static_cast<float>(per_order[1] > 0 ? per_order_sum[1] / per_order[1] : 0.0),
            per_order_min[2] < 1e8f ? per_order_min[2] : 0.0f, static_cast<float>(per_order[2] > 0 ? per_order_sum[2] / per_order[2] : 0.0),
            per_order_min[3] < 1e8f ? per_order_min[3] : 0.0f, static_cast<float>(per_order[3] > 0 ? per_order_sum[3] / per_order[3] : 0.0),
            tips_checked[0], tips_checked[1], tips_checked[2], tips_checked[3],
            tip_dev_worst[0], tip_dev_worst[1], tip_dev_worst[2], tip_dev_worst[3]);
        rhizome::RhizomeLog("PRINT", line);
    }
}


#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace rhizome {

// ============================================================================
// 5. CUDA GPU 3D Kelp Marine FSPM Growth Engine
//    (11-Feeler Cone Sensing, Spore Pruning, Developmental FSPM Branching)
// ============================================================================

struct CudaTipProposal {
    int action;        // 0 = sleeping, -1 = stalled, 1 = advance
    glm::vec3 best_p1;
    glm::vec3 best_dir;
    float r1;
    int next_depth;
    int sprout_lateral; // 1 = sprout lateral branch
    glm::vec3 lat_dir;
    float lat_r;
};

#ifdef __CUDACC__

__device__ void dev_closest_points_segment_segment(
    const glm::vec3& p1, const glm::vec3& q1,
    const glm::vec3& p2, const glm::vec3& q2,
    float& s, float& t,
    glm::vec3& c1, glm::vec3& c2
) {
    glm::vec3 d1 = q1 - p1;
    glm::vec3 d2 = q2 - p2;
    glm::vec3 r = p1 - p2;
    float a = glm::dot(d1, d1);
    float e = glm::dot(d2, d2);
    float f = glm::dot(d2, r);

    if (a <= 1e-8f && e <= 1e-8f) {
        s = t = 0.0f;
        c1 = p1;
        c2 = p2;
        return;
    }
    if (a <= 1e-8f) {
        s = 0.0f;
        t = dev_clamp(f / e, 0.0f, 1.0f);
    } else {
        float c = glm::dot(d1, r);
        if (e <= 1e-8f) {
            t = 0.0f;
            s = dev_clamp(-c / a, 0.0f, 1.0f);
        } else {
            float b = glm::dot(d1, d2);
            float denom = a * e - b * b;
            if (denom != 0.0f) {
                s = dev_clamp((b * f - c * e) / denom, 0.0f, 1.0f);
            } else {
                s = 0.0f;
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = dev_clamp(-c / a, 0.0f, 1.0f);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = dev_clamp((b - c) / a, 0.0f, 1.0f);
            }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

__device__ bool dev_shares_recent_fork(
    const CudaVolSection* __restrict__ sections,
    int sec_a,
    int sec_b,
    int max_hops
) {
    if (sec_a < 0 || sec_b < 0) return false;
    if (sec_a == sec_b) return true;
    // Same root strand: sections along own spine within depth distance are not external obstacles
    if (sections[sec_a].root_id == sections[sec_b].root_id) {
        int da = sections[sec_a].depth;
        int db = sections[sec_b].depth;
        int diff = (da > db) ? (da - db) : (db - da);
        if (diff <= max_hops) return true;
    }
    int anc_a[24];
    int count_a = 0;
    int cur = sec_a;
    while (cur >= 0 && count_a <= max_hops && count_a < 24) {
        anc_a[count_a++] = cur;
        cur = sections[cur].parent_sec;
    }
    cur = sec_b;
    int hops_b = 0;
    while (cur >= 0 && hops_b <= max_hops && hops_b < 24) {
        for (int k = 0; k < count_a; ++k) {
            if (anc_a[k] == cur) return true;
        }
        cur = sections[cur].parent_sec;
        hops_b++;
    }
    return false;
}

// 3D physical obstacle clearance sensor for the developmental FSPM engine.
// Tests true capsule-to-capsule surface clearance against foreign roots, while
// properly handling branch forks, own root spine, and lateral branch junctions.
// (Historic note: written for mode 6; since v24.16 it also serves the Kelp
// Marine FSPM growth kernel unconditionally.)
__device__ float dev_sense_3d_root_obstacles_dev3d(
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t hash_mask,
    float inv_body_cell,
    float step_len,
    int collar_hops,
    const glm::vec3& p0,
    const glm::vec3& p1,
    float r0,
    float r1,
    int parent_sec_idx,
    int my_root_id,
    glm::vec3* out_surface_repel,
    bool is_lateral_probe = false
) {
    glm::vec3 mid = (p0 + p1) * 0.5f;
    int bcx = static_cast<int>(floorf(mid.x * inv_body_cell));
    int bcy = static_cast<int>(floorf(mid.y * inv_body_cell));
    int bcz = static_cast<int>(floorf(mid.z * inv_body_cell));

    float min_margin = 1e9f;
    glm::vec3 repel_sum(0.0f);
    const float min_print_gap = 0.06f; // physical clearance in mm ensuring no volumetric intersection

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                int qx = bcx + dx, qy = bcy + dy, qz = bcz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, hash_mask);
                for (int sec_j = body_head[slot]; sec_j >= 0; sec_j = body_next[sec_j]) {
                    glm::ivec3 sc = body_cells[sec_j];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;

                    CudaVolSection sj = sections[sec_j];
                    if (sj.parent_sec < 0) continue;

                    // If lateral probe, skip the parent section and its direct parent where the branch emerges
                    if (is_lateral_probe && (sec_j == parent_sec_idx || sj.parent_sec == parent_sec_idx)) {
                        continue;
                    }

                    // Check if sec_j is our own parent, spine, or recent collar sibling
                    bool is_sibling = dev_shares_recent_fork(sections, parent_sec_idx, sec_j, collar_hops);
                    if (is_sibling) {
                        glm::vec3 diff_sib = p1 - sj.pos;
                        float d_sib = sqrtf(glm::dot(diff_sib, diff_sib));
                        if (d_sib > 1e-5f && d_sib < (r1 + sj.radius) * 1.5f) {
                            repel_sum += (diff_sib / d_sib) * 0.40f;
                        }
                        continue;
                    }

                    CudaVolSection spj = sections[sj.parent_sec];
                    float s = 0.0f, t = 0.0f;
                    glm::vec3 c1(0.0f), c2(0.0f);
                    dev_closest_points_segment_segment(p0, p1, spj.pos, sj.pos, s, t, c1, c2);

                    float rc1 = r0 + (r1 - r0) * s;
                    float rc2 = spj.radius + (sj.radius - spj.radius) * t;

                    glm::vec3 diff = c1 - c2;
                    float center_dist = sqrtf(glm::dot(diff, diff));
                    float surface_gap = center_dist - (rc1 + rc2);

                    float margin = surface_gap - min_print_gap;
                    if (margin < min_margin) {
                        min_margin = margin;
                    }

                    // Accumulate deflection force pointing outward away from obstacle
                    float sense_dist = min_print_gap + step_len * 1.25f;
                    if (surface_gap < sense_dist && center_dist > 1e-5f) {
                        float w = (sense_dist - surface_gap) / sense_dist;
                        repel_sum += (diff / center_dist) * (w * w);
                    }
                }
            }
        }
    }

    if (out_surface_repel) *out_surface_repel = repel_sum;
    return min_margin;
}

// Kernel 1: Insert all 3D root sections into GPU spatial hash table
__global__ void gpu_insert_sections_hash_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    float inv_cell_size,
    uint32_t hash_mask,
    int* __restrict__ body_head,
    int* __restrict__ body_next,
    glm::ivec3* __restrict__ body_cells
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int p = sections[i].parent_sec;
    glm::vec3 mid = (p >= 0) ? ((sections[p].pos + sections[i].pos) * 0.5f) : sections[i].pos;
    int cx = static_cast<int>(floorf(mid.x * inv_cell_size));
    int cy = static_cast<int>(floorf(mid.y * inv_cell_size));
    int cz = static_cast<int>(floorf(mid.z * inv_cell_size));
    body_cells[i] = glm::ivec3(cx, cy, cz);

    uint32_t slot = dev_hash_coords(cx, cy, cz, hash_mask);
    int prev = atomicExch(&body_head[slot], i);
    body_next[i] = prev;
}

// Kernel 2: Prune spores occupied by 3D root bodies and insert surviving spores into GPU spatial hash table
__global__ void gpu_prune_and_hash_spores_kernel(
    const glm::vec3* __restrict__ spores,
    int num_spores,
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t body_hash_mask,
    float inv_body_cell,
    float kill_radius,
    float inv_spore_cell,
    uint32_t spore_hash_mask,
    uint8_t* __restrict__ spore_alive,
    int* __restrict__ spore_head,
    int* __restrict__ spore_next,
    glm::ivec3* __restrict__ spore_cells
) {
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (s_idx >= num_spores) return;

    glm::vec3 sp = spores[s_idx];
    int bcx = static_cast<int>(floorf(sp.x * inv_body_cell));
    int bcy = static_cast<int>(floorf(sp.y * inv_body_cell));
    int bcz = static_cast<int>(floorf(sp.z * inv_body_cell));

    bool occupied = false;
    for (int dx = -1; dx <= 1 && !occupied; ++dx) {
        for (int dy = -1; dy <= 1 && !occupied; ++dy) {
            for (int dz = -1; dz <= 1 && !occupied; ++dz) {
                int qx = bcx + dx, qy = bcy + dy, qz = bcz + dz;
                uint32_t slot = dev_hash_coords(qx, qy, qz, body_hash_mask);
                for (int sec_i = body_head[slot]; sec_i >= 0; sec_i = body_next[sec_i]) {
                    glm::ivec3 sc = body_cells[sec_i];
                    if (sc.x != qx || sc.y != qy || sc.z != qz) continue;

                    CudaVolSection sec = sections[sec_i];
                    // Frozen root sections (pinned bit 0 set) do not kill spores or starve new growing roots
                    if ((sec.pinned & 1) != 0) continue;

                    float d_surf = 0.0f;
                    if (sec.parent_sec >= 0) {
                        CudaVolSection psec = sections[sec.parent_sec];
                        glm::vec3 ba = sec.pos - psec.pos;
                        float l2 = glm::dot(ba, ba);
                        float t = (l2 > 1e-8f) ? dev_clamp(glm::dot(sp - psec.pos, ba) / l2, 0.0f, 1.0f) : 0.0f;
                        glm::vec3 proj = psec.pos + t * ba;
                        float r_at_t = psec.radius + (sec.radius - psec.radius) * t;
                        glm::vec3 dvec = sp - proj;
                        d_surf = sqrtf(glm::dot(dvec, dvec)) - r_at_t;
                    } else {
                        glm::vec3 dvec = sp - sec.pos;
                        d_surf = sqrtf(glm::dot(dvec, dvec)) - sec.radius;
                    }
                    if (d_surf <= kill_radius) {
                        occupied = true;
                        break;
                    }
                }
            }
        }
    }

    if (occupied) {
        spore_alive[s_idx] = 0;
        spore_next[s_idx] = -1;
    } else {
        spore_alive[s_idx] = 1;
        int scx = static_cast<int>(floorf(sp.x * inv_spore_cell));
        int scy = static_cast<int>(floorf(sp.y * inv_spore_cell));
        int scz = static_cast<int>(floorf(sp.z * inv_spore_cell));
        spore_cells[s_idx] = glm::ivec3(scx, scy, scz);
        uint32_t slot = dev_hash_coords(scx, scy, scz, spore_hash_mask);
        int prev = atomicExch(&spore_head[slot], s_idx);
        spore_next[s_idx] = prev;
    }
}

// Kernel 3: Parallel GPU 11-Feeler Cone Environment Sensing & Lateral Branch Probing (1 GPU thread per active root tip)

__global__ void gpu_compact_surviving_spores_vol_kernel(
    const glm::vec3* __restrict__ in_spores,
    const uint8_t* __restrict__ spore_alive,
    int num_spores,
    glm::vec3* __restrict__ out_spores,
    int* __restrict__ out_surviving_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;
    if (spore_alive[i]) {
        int slot = atomicAdd(out_surviving_count, 1);
        out_spores[slot] = in_spores[i];
    }
}


__global__ void gpu_init_volumetric_seeds_kernel(
    const glm::vec3* __restrict__ seeds,
    int num_seeds,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_vessel_sdf,
    float base_radius,
    uint32_t rng_seed,
    float wall_clearance_scale,
    CudaVolSection* __restrict__ out_sections,
    CudaVolTip* __restrict__ out_tips,
    int primary_roots
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;

    CudaVolSection seed_sec;
    seed_sec.pos = seeds[s];
    seed_sec.radius = base_radius;
    seed_sec.parent_sec = -1;
    seed_sec.root_id = s;
    seed_sec.depth = 0;
    seed_sec.pinned = 1;

    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        voxel_size, min_vessel_sdf,
        seed_sec.pos, seed_sec.radius, nullptr,
        wall_clearance_scale
    );
    out_sections[s] = seed_sec;

    glm::vec3 wall_n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, seed_sec.pos);
    glm::vec3 up = (fabsf(wall_n.z) < 0.85f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t1 = dev_safe_norm(glm::cross(wall_n, up), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 t2 = dev_safe_norm(glm::cross(wall_n, t1), glm::vec3(0.0f, 1.0f, 0.0f));

    const int primary_count = (primary_roots > 0) ? primary_roots : 3;
    for (int k = 0; k < primary_count; ++k) {
        float u = dev_hash_01(rng_seed + static_cast<uint32_t>(s * 17 + k * 31));
        float ang = (static_cast<float>(k) + 0.25f * u) * (2.0f * 3.14159265f / static_cast<float>(primary_count));
        glm::vec3 tangent_dir = cosf(ang) * t1 + sinf(ang) * t2;
        glm::vec3 inward_dir = -wall_n * 0.45f + tangent_dir * 0.85f;
        glm::vec3 d = dev_safe_norm(inward_dir, tangent_dir);
        CudaVolTip tip;
        tip.sec_idx = s;
        tip.root_id = num_seeds + s * primary_count + k;
        tip.dir = d;
        tip.radius = base_radius;
        tip.steps_since_branch = k;
        tip.stalled_steps = 0;
        tip.generation = 0;
        tip.steps_grown = 0;
        out_tips[s * primary_count + k] = tip;
    }
}

__global__ void gpu_convert_vol_sections_to_nodes_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    GpuColonizationNodeLayout* __restrict__ out_nodes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    GpuColonizationNodeLayout cn;
    cn.position = sections[i].pos;
    cn.parent_index = sections[i].parent_sec;
    cn.radius = sections[i].radius;
    cn.depth = sections[i].depth;
    cn.energy = 1.0f;
    cn.strand_id = sections[i].root_id;
    out_nodes[i] = cn;
}

__global__ void gpu_check_awake_tips_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    int* __restrict__ out_awake_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    if (tips[i].stalled_steps <= 4) {
        atomicAdd(out_awake_count, 1);
    }
}

__global__ void gpu_reset_stalled_tips_kernel(
    CudaVolTip* __restrict__ tips,
    int num_tips
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    if (tips[i].stalled_steps < 1000) {
        tips[i].stalled_steps = 0;
        tips[i].steps_grown = 0;
    }
}

__global__ void gpu_mark_section_children_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int* __restrict__ out_has_child
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    int p = sections[i].parent_sec;
    if (p >= 0 && p < num_sections) {
        out_has_child[p] = 1;
    }
}

__global__ void gpu_mark_active_tip_sections_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    int num_sections,
    int* __restrict__ out_has_child
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    int sec = tips[i].sec_idx;
    if (sec >= 0 && sec < num_sections) {
        out_has_child[sec] = 1;
    }
}

__global__ void gpu_respawn_leaf_tips_kernel(
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ has_child,
    int num_sections,
    CudaVolTip* __restrict__ io_tips,
    int* __restrict__ d_tip_count,
    int max_tips
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sections) return;
    int i = (num_sections - 1) - idx;
    if (has_child[i] == 0 && sections[i].parent_sec >= 0 && (sections[i].pinned & 1) == 0) {
        glm::vec3 d = sections[i].pos - sections[sections[i].parent_sec].pos;
        float dl = sqrtf(glm::dot(d, d));
        if (dl > 1e-5f) {
            int slot = atomicAdd(d_tip_count, 1);
            if (slot < max_tips) {
                CudaVolTip tip;
                tip.sec_idx = i;
                tip.root_id = sections[i].root_id;
                tip.dir = d / dl;
                tip.radius = sections[i].radius;
                tip.steps_since_branch = 2;
                tip.stalled_steps = 0;
                tip.generation = max(1, sections[i].pinned >> 1);
                tip.steps_grown = 0;
                io_tips[slot] = tip;
            }
        }
    }
}

__device__ inline int dev_find_vol_section_seed_slot(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int idx
) {
    int cur = idx;
    for (int hop = 0; hop < 4096; ++hop) {
        int p = sections[cur].parent_sec;
        if (p < 0 || p >= num_sections || p == cur) break;
        cur = p;
    }
    return sections[cur].root_id;
}

__global__ void gpu_sync_volumetric_freeze_sections_kernel(
    CudaVolSection* __restrict__ sections,
    int num_sections,
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    int num_seeds,
    int* __restrict__ out_seed_exists
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, i);
    if (sections[i].parent_sec < 0 && seed_slot >= 0 && seed_slot < num_seeds) {
        out_seed_exists[seed_slot] = 1;
        // Keep seed root section position in sync with moved seed position
        if (frozen_seeds == nullptr || frozen_seeds[seed_slot] == 0) {
            sections[i].pos = seeds[seed_slot];
        }
    }
    bool is_frozen = (seed_slot >= 0 && seed_slot < num_seeds && frozen_seeds != nullptr && frozen_seeds[seed_slot] != 0);
    if (sections[i].parent_sec < 0) {
        sections[i].pinned = is_frozen ? 3 : 1;
    } else if (is_frozen) {
        sections[i].pinned |= 1;
    } else {
        sections[i].pinned &= ~1;
    }
}

__global__ void gpu_sync_volumetric_freeze_tips_kernel(
    CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    const uint8_t* __restrict__ frozen_seeds,
    int num_seeds
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;
    int sec = tips[i].sec_idx;
    if (sec < 0 || sec >= num_sections) return;
    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, sec);
    bool is_frozen = (seed_slot >= 0 && seed_slot < num_seeds && frozen_seeds != nullptr && frozen_seeds[seed_slot] != 0);
    if (is_frozen) {
        tips[i].stalled_steps = 10000;
    } else if (tips[i].stalled_steps >= 1000) {
        tips[i].stalled_steps = 0;
    }
}

__global__ void gpu_spawn_missing_volumetric_seeds_kernel(
    const glm::vec3* __restrict__ seeds,
    const uint8_t* __restrict__ frozen_seeds,
    const int* __restrict__ seed_exists,
    int num_seeds,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_vessel_sdf,
    float base_radius,
    uint32_t rng_seed,
    float wall_clearance_scale,
    CudaVolSection* __restrict__ io_sections,
    int* __restrict__ d_num_sections,
    int max_sections,
    CudaVolTip* __restrict__ io_tips,
    int* __restrict__ d_num_tips,
    int max_tips,
    int* __restrict__ d_next_root_id,
    int* __restrict__ d_spawned_count,
    int primary_roots
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;
    if (seed_exists[s] != 0) return;

    int sec_idx = atomicAdd(d_num_sections, 1);
    if (sec_idx >= max_sections) return;

    bool is_frozen = (frozen_seeds != nullptr && frozen_seeds[s] != 0);

    CudaVolSection seed_sec;
    seed_sec.pos = seeds[s];
    seed_sec.radius = base_radius;
    seed_sec.parent_sec = -1;
    seed_sec.root_id = s;
    seed_sec.depth = 0;
    seed_sec.pinned = is_frozen ? 3 : 1;

    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        voxel_size, min_vessel_sdf,
        seed_sec.pos, seed_sec.radius, nullptr,
        wall_clearance_scale
    );
    io_sections[sec_idx] = seed_sec;
    atomicAdd(d_spawned_count, 1);

    glm::vec3 wall_n = dev_sample_sdf_grad(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, seed_sec.pos);
    glm::vec3 up = (fabsf(wall_n.z) < 0.85f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 t1 = dev_safe_norm(glm::cross(wall_n, up), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::vec3 t2 = dev_safe_norm(glm::cross(wall_n, t1), glm::vec3(0.0f, 1.0f, 0.0f));

    const int primary_count = (primary_roots > 0) ? primary_roots : 3;
    for (int k = 0; k < primary_count; ++k) {
        int tip_slot = atomicAdd(d_num_tips, 1);
        if (tip_slot >= max_tips) break;
        float u = dev_hash_01(rng_seed + static_cast<uint32_t>(s * 17 + k * 31));
        float ang = (static_cast<float>(k) + 0.25f * u) * (2.0f * 3.14159265f / static_cast<float>(primary_count));
        glm::vec3 tangent_dir = cosf(ang) * t1 + sinf(ang) * t2;
        glm::vec3 inward_dir = -wall_n * 0.45f + tangent_dir * 0.85f;
        glm::vec3 d = dev_safe_norm(inward_dir, tangent_dir);
        CudaVolTip tip;
        tip.sec_idx = sec_idx;
        tip.root_id = atomicAdd(d_next_root_id, 1);
        tip.dir = d;
        tip.radius = base_radius;
        tip.steps_since_branch = k;
        tip.stalled_steps = is_frozen ? 10000 : 0;
        tip.generation = 0;
        tip.steps_grown = 0;
        io_tips[tip_slot] = tip;
    }
}

// =========================================================================
// v24.16 — DEVELOPMENTAL 3D VOLUMETRIC FSPM (the Kelp growth brain)
// Developmental FSPM growth (Firn-Digby gravitropic sine law setpoints,
// per-order elongation/spacing/length caps, acropetal lateral emergence),
// with DIRECT 3D VOLUME AWARENESS and COLLISION SENSING.
// Evaluates 3D capsule distance to existing root volume via the spatial hash,
// uses feeler probing and tangent deflection to steer around other roots,
// and halts tips if blocked to completely prevent root-to-root overlap.
// =========================================================================
__global__ void gpu_dev3d_advance_tips_kernel(
    const CudaVolTip* __restrict__ tips,
    int num_tips,
    const CudaVolSection* __restrict__ sections,
    const int* __restrict__ body_head,
    const int* __restrict__ body_next,
    const glm::ivec3* __restrict__ body_cells,
    uint32_t body_hash_mask,
    float inv_body_cell,
    const glm::vec3* __restrict__ spores,
    const int* __restrict__ spore_head,
    const int* __restrict__ spore_next,
    const glm::ivec3* __restrict__ spore_cells,
    uint32_t spore_hash_mask,
    float inv_spore_cell,
    int num_spores,
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    CudaVolGrowthParams params,
    int collar_hops,
    float min_skin_gap,
    CudaTipProposal* __restrict__ out_proposals
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = tips[ti];
    CudaTipProposal prop;
    prop.action = 0;
    prop.best_p1 = glm::vec3(0.0f);
    prop.best_dir = tip.dir;
    prop.r1 = tip.radius;
    prop.next_depth = 0;
    prop.sprout_lateral = 0;
    prop.lat_dir = glm::vec3(0.0f);
    prop.lat_r = fmaxf(params.dev_min_print_radius, 0.10f);

    // Give tips up to 12 attempts to navigate around obstacles or follow spores before parking
    if (tip.stalled_steps > 12) {
        out_proposals[ti] = prop;
        return;
    }

    const int order = min(max(tip.generation, 0), 3);
    const glm::vec3 p0 = sections[tip.sec_idx].pos;
    const float r0 = sections[tip.sec_idx].radius;
    const int next_depth = sections[tip.sec_idx].depth + 1;

    int steps_grown = tip.steps_grown;
    if (steps_grown < 0) steps_grown = 0;
    if (steps_grown >= params.dev_max_steps[order]) {
        prop.action = -1;
        out_proposals[ti] = prop;
        return;
    }

    const float r1 = fmaxf(params.dev_order_radius[order], params.dev_min_print_radius);
    float step_len = fmaxf(0.25f, params.growth_step * params.dev_rate[order]);
    // v24.21 Surface Density (CTRL256): the node spacing floor as a user
    // lever — every newly placed node keeps at least (tip radius x this)
    // distance. 1.12 = the shipped watertight guard (default, bit-for-bit);
    // the host config clamps to [0.95..1.60], re-clamped here so the kernel
    // can never see an out-of-range multiplier.
    step_len = fmaxf(step_len, r1 * dev_clamp(params.dev_surface_step_mult, 0.95f, 1.60f));

    glm::vec3 dir = tip.dir;

    // 1. Environmental Nutrient Spore Sensing (GPU Spatial Hash)
    // Pulls root tips into open, unvisited space inside the vessel
    glm::vec3 spore_pull(0.0f);
    int spore_pull_count = 0;
    if (num_spores > 0 && spores != nullptr && spore_head != nullptr) {
        const float search_r = fmaxf(step_len * 3.0f, params.search_radius);
        const float search_r2 = search_r * search_r;
        int scx = static_cast<int>(floorf(p0.x * inv_spore_cell));
        int scy = static_cast<int>(floorf(p0.y * inv_spore_cell));
        int scz = static_cast<int>(floorf(p0.z * inv_spore_cell));

        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int qx = scx + dx, qy = scy + dy, qz = scz + dz;
                    uint32_t slot = dev_hash_coords(qx, qy, qz, spore_hash_mask);
                    for (int s_idx = spore_head[slot]; s_idx >= 0; s_idx = spore_next[s_idx]) {
                        glm::ivec3 sc = spore_cells[s_idx];
                        if (sc.x != qx || sc.y != qy || sc.z != qz) continue;
                        glm::vec3 diff = spores[s_idx] - p0;
                        float d2 = glm::dot(diff, diff);
                        if (d2 < search_r2 && d2 > 1e-5f) {
                            float d = sqrtf(d2);
                            glm::vec3 sdir = diff / d;
                            float fwd = fmaxf(0.10f, 0.60f + 0.40f * glm::dot(sdir, tip.dir));
                            spore_pull += sdir * (fwd / (1.0f + d * 0.15f));
                            spore_pull_count++;
                        }
                    }
                }
            }
        }
    }

    if (spore_pull_count > 0 && glm::dot(spore_pull, spore_pull) > 1e-8f) {
        glm::vec3 norm_spore = dev_safe_norm(spore_pull, dir);
        float spore_w = dev_clamp(0.25f + 0.15f * fminf(static_cast<float>(spore_pull_count) / 10.0f, 1.0f), 0.15f, 0.50f);
        dir = dev_safe_norm(dir * (1.0f - spore_w) + norm_spore * spore_w, dir);
    }

    // 2. Directional wander
    if (params.dev_noise > 0.001f) {
        const float noise_eff = params.dev_noise * (order == 0 ? 0.45f : 1.0f);
        uint32_t salt = params.seed + static_cast<uint32_t>(tip.sec_idx * 37 + params.current_iter * 101);
        float nx = dev_hash_01(salt) * 2.0f - 1.0f;
        float ny = dev_hash_01(salt + 733u) * 2.0f - 1.0f;
        float nz = dev_hash_01(salt + 1531u) * 2.0f - 1.0f;
        glm::vec3 jitter(nx, ny, nz);
        jitter -= glm::dot(jitter, dir) * dir;
        if (glm::dot(jitter, jitter) > 1e-10f) {
            dir = dev_safe_norm(dir + dev_safe_norm(jitter, dir) * (noise_eff * 3.0f), dir);
        }
    }

    // 2. Firn-Digby gravitropic setpoint chase (sine law)
    // Mode 5 (Roots): gravitropism pulls downward toward soil (axis = (0, 0, -1)).
    // Mode 6 (Kelp Marine FSPM): positive buoyancy / phototropism pulls UPWARD toward sunlight/water surface (axis = (0, 0, 1))
    // with hydrodynamic ocean current drift and wave surge.
    const glm::vec3 tropism_axis(0.0f, 0.0f, 1.0f); // Kelp: upward buoyancy toward the surface
    const float effective_grav_gain = (params.dev_grav_gain < 1e-5f)
        ? 0.15f // Natural floating buoyancy response for marine algae
        : params.dev_grav_gain;

    if (effective_grav_gain > 1e-6f) {
        const float gsa_deg = fminf(fmaxf(params.dev_gsa_deg[order], 0.0f), 89.0f);
        const float gsa_rad = gsa_deg * (3.14159265f / 180.0f);
        glm::vec3 horiz = dir - tropism_axis * glm::dot(dir, tropism_axis);
        glm::vec3 plane1;
        if (glm::dot(horiz, horiz) > 1e-8f) {
            plane1 = glm::normalize(horiz);
        } else {
            float az = dev_hash_01(params.seed + static_cast<uint32_t>(tip.root_id * 40503u)) * 6.2831853f;
            plane1 = glm::vec3(cosf(az), sinf(az), 0.0f);
        }
        glm::vec3 set_dir = tropism_axis * cosf(gsa_rad) + plane1 * sinf(gsa_rad);

        // Hydrodynamic ocean current drift & wave surge for Kelp
        {
            float wave_time = static_cast<float>(params.current_iter) * 0.14f;
            float current_angle = 0.85f + 0.30f * sinf(wave_time);
            glm::vec3 current_vec(cosf(current_angle) * 0.20f, sinf(current_angle) * 0.15f, 0.0f);
            set_dir = dev_safe_norm(set_dir + current_vec, set_dir);
        }

        float cosang = fminf(fmaxf(glm::dot(dir, set_dir), -1.0f), 1.0f);
        float ang = acosf(cosang);
        if (ang > 1e-4f) {
            float turn = effective_grav_gain * sinf(ang);
            if (turn > ang) turn = ang;
            glm::vec3 axis = glm::cross(dir, set_dir);
            float ax2 = glm::dot(axis, axis);
            if (ax2 > 1e-12f) {
                axis *= rsqrtf(ax2);
                glm::vec3 rotated = dir * cosf(turn)
                    + glm::cross(axis, dir) * sinf(turn)
                    + axis * (glm::dot(axis, dir) * (1.0f - cosf(turn)));
                dir = dev_safe_norm(rotated, dir);
            } else {
                dir = set_dir;
            }
        }
    }

    // 3. Proposed developmental step (Firn-Digby sine law toward GSA setpoint)
    glm::vec3 p1 = p0 + dir * step_len;
    glm::vec3 intended = dir;
    glm::vec3 odir = dir;
    dev_confine_sphere_to_vessel(
        vessel_sdf, vessel_min_coord, vessel_dims,
        params.voxel_size, params.min_vessel_sdf,
        p1, r1, &odir, params.wall_clearance_scale
    );
    glm::vec3 moved = p1 - p0;
    float moved_len = sqrtf(glm::dot(moved, moved));
    if (moved_len < step_len * 0.25f) {
        prop.action = -1;
        out_proposals[ti] = prop;
        return;
    }
    dir = dev_safe_norm(moved * (1.0f / moved_len), dir);

    // Check wall-slide: only park if directly grinding into a hard wall (>75 deg)
    {
        float cos_defl = fminf(fmaxf(glm::dot(dir, intended), -1.0f), 1.0f);
        if (acosf(cos_defl) > 1.309f) { // 75 degrees in radians
            prop.action = -1;
            out_proposals[ti] = prop;
            return;
        }
    }

    // Sense 3D root volume obstacles
    glm::vec3 repel(0.0f);
    float margin = dev_sense_3d_root_obstacles_dev3d(
        sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
        step_len, collar_hops, p0, p1, r0, r1, tip.sec_idx, tip.root_id, &repel, false
    );

    if (margin < 0.0f) {
        // Physical obstacle detected: steer and glide around obstacle using multi-angle feelers!
        glm::vec3 best_glide_p1 = p1;
        glm::vec3 best_glide_dir = dir;
        float best_glide_score = -1e9f;
        bool found_clear_path = false;

        glm::vec3 repel_dir = (glm::dot(repel, repel) > 1e-8f)
            ? dev_safe_norm(repel, glm::vec3(0.0f, 1.0f, 0.0f))
            : glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 side_tangent = dev_safe_norm(glm::cross(dir, repel_dir), glm::vec3(1.0f, 0.0f, 0.0f));

        // Test feelers in a deflection cone around the obstacle
        // (angles tilted along obstacle tangent and outward normal)
        for (int f = 0; f < 8; ++f) {
            float ang = static_cast<float>(f) * 0.785398f;
            float tilt = (tip.stalled_steps > 2) ? 0.75f : 0.45f;
            glm::vec3 cand_dir = dev_safe_norm(
                dir * (1.0f - tilt) + (cosf(ang) * repel_dir + sinf(ang) * side_tangent) * tilt,
                dir
            );
            glm::vec3 cand_p1 = p0 + cand_dir * step_len;
            glm::vec3 cand_odir = cand_dir;
            dev_confine_sphere_to_vessel(
                vessel_sdf, vessel_min_coord, vessel_dims,
                params.voxel_size, params.min_vessel_sdf,
                cand_p1, r1, &cand_odir, params.wall_clearance_scale
            );
            glm::vec3 cm = cand_p1 - p0;
            float clen = sqrtf(glm::dot(cm, cm));
            if (clen < step_len * 0.25f) continue;
            cand_dir = cm * (1.0f / clen);

            float cand_margin = dev_sense_3d_root_obstacles_dev3d(
                sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
                step_len, collar_hops, p0, cand_p1, r0, r1, tip.sec_idx, tip.root_id, nullptr, false
            );

            if (cand_margin >= 0.0f) {
                // Direction clears all root bodies!
                float score = cand_margin * 2.0f + glm::dot(cand_dir, intended) * 1.5f;
                if (spore_pull_count > 0) {
                    score += glm::dot(cand_dir, dev_safe_norm(spore_pull, cand_dir)) * 1.0f;
                }
                if (score > best_glide_score) {
                    best_glide_score = score;
                    best_glide_p1 = cand_p1;
                    best_glide_dir = cand_dir;
                    found_clear_path = true;
                }
            }
        }

        if (found_clear_path) {
            p1 = best_glide_p1;
            dir = best_glide_dir;
        } else {
            // Truly trapped on all angles: stall on this step
            prop.action = -1;
            out_proposals[ti] = prop;
            return;
        }
    }

    prop.action = 1;
    prop.best_p1 = p1;
    prop.best_dir = dir;
    prop.r1 = r1;
    prop.next_depth = next_depth;

    // 5. Acropetal lateral emergence with physical 3D clearance check
    uint32_t branch_prob_hash = params.seed + static_cast<uint32_t>(tip.root_id * 17923u + params.current_iter * 997u + tip.sec_idx * 53u);
    float branch_roll = dev_hash_01(branch_prob_hash);
    if (tip.generation < 3 &&
        (tip.steps_since_branch + 1) >= params.dev_spacing[order] &&
        (steps_grown + 1) < params.dev_max_steps[order] &&
        branch_roll <= fminf(fmaxf(params.branching_rate, 0.05f), 1.0f)) {
        const int child_order = order + 1;
        glm::vec3 ref = (fabsf(dir.z) < 0.9f) ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 perp1 = dev_safe_norm(glm::cross(dir, ref), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 perp2 = glm::cross(dir, perp1);
        uint32_t az_hash = params.seed + static_cast<uint32_t>(tip.root_id * 40503u + params.current_iter * 613u + tip.sec_idx * 83u);
        float az = dev_hash_01(az_hash) * 6.2831853f;
        float ba = fminf(fmaxf(params.dev_branch_angle, 15.0f), 89.0f) * (3.14159265f / 180.0f);
        const float lat_r = fmaxf(params.dev_order_radius[child_order], params.dev_min_print_radius);
        float lat_step = fmaxf(0.25f, step_len * 0.6f);

        for (int try_az = 0; try_az < 4; ++try_az) {
            float cur_az = az + static_cast<float>(try_az) * 1.570796f;
            glm::vec3 radial_dir = perp1 * cosf(cur_az) + perp2 * sinf(cur_az);
            glm::vec3 lat_dir = dev_safe_norm(dir * cosf(ba) + radial_dir * sinf(ba), dir);
            glm::vec3 lat_p1 = p1 + lat_dir * lat_step;
            glm::vec3 ldir = lat_dir;
            dev_confine_sphere_to_vessel(
                vessel_sdf, vessel_min_coord, vessel_dims,
                params.voxel_size, params.min_vessel_sdf,
                lat_p1, lat_r, &ldir, params.wall_clearance_scale
            );
            glm::vec3 lm = lat_p1 - p1;
            if (sqrtf(glm::dot(lm, lm)) >= step_len * 0.20f) {
                float lat_margin = dev_sense_3d_root_obstacles_dev3d(
                    sections, body_head, body_next, body_cells, body_hash_mask, inv_body_cell,
                    lat_step, collar_hops, p1, lat_p1, r1, lat_r, tip.sec_idx, tip.root_id, nullptr, true
                );
                if (lat_margin >= 0.0f) {
                    prop.sprout_lateral = 1;
                    prop.lat_dir = dev_safe_norm(lm, lat_dir);
                    prop.lat_r = lat_r;
                    break;
                }
            }
        }
    }

    out_proposals[ti] = prop;
}

// v24.12 Track F R1: developmental commit — the shared append discipline
// (conflict check against lower-index tips, atomic section append, lateral
// child creation) plus steps_grown bookkeeping the shared kernel does not
// carry. Stand-off for the same-step conflict check is a fixed 10% + 0.1 mm
// (no min_skin_gap coupling: developmental roots may touch and graft — the
// SDF bake unions them into one printable solid).
__global__ void gpu_dev_commit_proposals_kernel(
    const CudaVolTip* __restrict__ in_tips,
    const CudaTipProposal* __restrict__ proposals,
    int num_tips,
    CudaVolSection* __restrict__ io_sections,
    int* __restrict__ d_num_sections,
    int max_sections,
    CudaVolTip* __restrict__ out_next_tips,
    int* __restrict__ d_num_next_tips,
    int max_next_tips,
    int* __restrict__ d_next_root_id
) {
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= num_tips) return;

    CudaVolTip tip = in_tips[ti];
    CudaTipProposal prop = proposals[ti];

    if (prop.action == 0) {
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }
    if (prop.action < 0) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    // Parallel conflict check against lower-index advancing tips in the same step
    bool conflict = false;
    for (int oj = 0; oj < ti; ++oj) {
        if (proposals[oj].action <= 0) continue;
        if (in_tips[oj].sec_idx == tip.sec_idx) continue;
        glm::vec3 d = prop.best_p1 - proposals[oj].best_p1;
        float min_d = (prop.r1 + proposals[oj].r1) * 1.10f + 0.10f;
        if (glm::dot(d, d) < min_d * min_d) {
            conflict = true;
            break;
        }
    }
    if (conflict) {
        tip.stalled_steps++;
        int slot = atomicAdd(d_num_next_tips, 1);
        if (slot < max_next_tips) out_next_tips[slot] = tip;
        return;
    }

    int new_sec_idx = atomicAdd(d_num_sections, 1);
    if (new_sec_idx >= max_sections) return;

    CudaVolSection new_sec;
    new_sec.pos = prop.best_p1;
    new_sec.radius = prop.r1;
    new_sec.parent_sec = tip.sec_idx;
    new_sec.root_id = tip.root_id;
    new_sec.depth = prop.next_depth;
    new_sec.pinned = (tip.generation << 1); // order storage for Murray/taper (existing convention)
    io_sections[new_sec_idx] = new_sec;

    tip.sec_idx = new_sec_idx;
    tip.dir = prop.best_dir;
    tip.radius = prop.r1;
    tip.steps_since_branch = prop.sprout_lateral ? 0 : (tip.steps_since_branch + 1);
    tip.stalled_steps = 0;
    tip.steps_grown = (tip.steps_grown < 0 ? 0 : tip.steps_grown) + 1;

    int slot_main = atomicAdd(d_num_next_tips, 1);
    if (slot_main < max_next_tips) {
        out_next_tips[slot_main] = tip;
    }

    if (prop.sprout_lateral) {
        CudaVolTip child_tip;
        child_tip.sec_idx = new_sec_idx;
        child_tip.root_id = atomicAdd(d_next_root_id, 1);
        child_tip.dir = prop.lat_dir;
        child_tip.radius = prop.lat_r;
        child_tip.steps_since_branch = 0;
        child_tip.stalled_steps = 0;
        child_tip.generation = tip.generation + 1;
        child_tip.steps_grown = 0;
        int slot_lat = atomicAdd(d_num_next_tips, 1);
        if (slot_lat < max_next_tips) {
            out_next_tips[slot_lat] = child_tip;
        }
    }
}

#endif // __CUDACC__

bool CudaRasterizer::step_volumetric_colonization_gpu(
    const CudaVolumeGrid& vessel_grid,
    const CudaVolGrowthParams& params,
    std::vector<glm::vec3>& io_spores,
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int& io_next_root_id
) {
#ifdef __CUDACC__
    if (!vessel_grid.is_valid() || io_sections.empty() || io_tips.empty()) {
        return false;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }

    // 1. Cache the 3D Vessel SDF grid in GPU VRAM across growth steps so it is only uploaded once per specimen
    static float* s_d_vessel_sdf = nullptr;
    static const float* s_cached_host_ptr = nullptr;
    static size_t s_cached_sdf_count = 0;
    static uint64_t s_cached_version = 0;

    if (s_cached_host_ptr != vessel_grid.data.data() || s_cached_sdf_count != vessel_grid.data.size() || s_cached_version != vessel_grid.version) {
        if (s_d_vessel_sdf) {
            cudaFree(s_d_vessel_sdf);
            s_d_vessel_sdf = nullptr;
        }
        size_t bytes = vessel_grid.data.size() * sizeof(float);
        if (cudaMalloc(&s_d_vessel_sdf, bytes) != cudaSuccess) {
            s_d_vessel_sdf = nullptr;
            return false;
        }
        cudaMemcpy(s_d_vessel_sdf, vessel_grid.data.data(), bytes, cudaMemcpyHostToDevice);
        s_cached_host_ptr = vessel_grid.data.data();
        s_cached_sdf_count = vessel_grid.data.size();
        s_cached_version = vessel_grid.version;
    }

    const float ref_step = params.variable_step
        ? std::max(params.min_growth_step, params.max_growth_step)
        : params.growth_step;
    const float step_len = std::max(0.25f, ref_step);
    const float search_r = std::max(step_len * 2.5f, params.search_radius);
    const float min_skin_gap = std::max(params.voxel_size * 1.85f, params.tip_radius * params.skin_gap_scale);
    const float seg_cell_size = std::max(step_len * 2.4f, (params.base_radius * 2.2f + min_skin_gap) * 1.6f);
    const float inv_body_cell = 1.0f / seg_cell_size;
    const float inv_spore_cell = 1.0f / search_r;

    const int collar_hops = 20;
    // v24.23 fix-round 2: removed dead local `branch_interval` (MSVC C4189).
    // It was computed from params.branching_rate but never referenced — the
    // growth kernel derives its own branching decision directly from
    // params.branching_rate (see branch_roll check). Pre-existing in v24.23;
    // verified single occurrence in the whole tree before removal.

    const uint32_t body_hash_size = 65536u;
    const uint32_t body_hash_mask = body_hash_size - 1u;
    const uint32_t spore_hash_size = 65536u;
    const uint32_t spore_hash_mask = spore_hash_size - 1u;

    // Persistent reusable GPU buffers to avoid per-step cudaMalloc overhead
    static CudaVolSection* s_d_sections = nullptr;
    static size_t s_cap_sections = 0;
    static int* s_d_body_head = nullptr;
    static int* s_d_body_next = nullptr;
    static glm::ivec3* s_d_body_cells = nullptr;

    static glm::vec3* s_d_spores = nullptr;
    static glm::vec3* s_d_spores_compact = nullptr;
    static size_t s_cap_spores = 0;
    static uint8_t* s_d_spore_alive = nullptr;
    static int* s_d_spore_head = nullptr;
    static int* s_d_spore_next = nullptr;
    static glm::ivec3* s_d_spore_cells = nullptr;

    static CudaVolTip* s_d_tips = nullptr;
    static CudaVolTip* s_d_next_tips = nullptr;
    static CudaTipProposal* s_d_proposals = nullptr;
    static size_t s_cap_tips = 0;
    static int* s_d_step_counters = nullptr; // [0]=surviving_spores, [1]=num_sections, [2]=num_next_tips, [3]=next_root_id

    if (!s_d_body_head) {
        cudaMalloc(&s_d_body_head, body_hash_size * sizeof(int));
        cudaMalloc(&s_d_spore_head, spore_hash_size * sizeof(int));
        cudaMalloc(&s_d_step_counters, 4 * sizeof(int));
    }

    size_t need_sections = std::max<size_t>(io_sections.size() + io_tips.size() * 2 + 256, 16384);
    if (need_sections > s_cap_sections) {
        if (s_d_sections) cudaFree(s_d_sections);
        if (s_d_body_next) cudaFree(s_d_body_next);
        if (s_d_body_cells) cudaFree(s_d_body_cells);
        s_cap_sections = need_sections * 2;
        cudaMalloc(&s_d_sections, s_cap_sections * sizeof(CudaVolSection));
        cudaMalloc(&s_d_body_next, s_cap_sections * sizeof(int));
        cudaMalloc(&s_d_body_cells, s_cap_sections * sizeof(glm::ivec3));
    }

    size_t need_spores = std::max<size_t>(io_spores.size() + 256, 16384);
    if (need_spores > s_cap_spores) {
        if (s_d_spores) cudaFree(s_d_spores);
        if (s_d_spores_compact) cudaFree(s_d_spores_compact);
        if (s_d_spore_alive) cudaFree(s_d_spore_alive);
        if (s_d_spore_next) cudaFree(s_d_spore_next);
        if (s_d_spore_cells) cudaFree(s_d_spore_cells);
        s_cap_spores = need_spores * 2;
        cudaMalloc(&s_d_spores, s_cap_spores * sizeof(glm::vec3));
        cudaMalloc(&s_d_spores_compact, s_cap_spores * sizeof(glm::vec3));
        cudaMalloc(&s_d_spore_alive, s_cap_spores * sizeof(uint8_t));
        cudaMalloc(&s_d_spore_next, s_cap_spores * sizeof(int));
        cudaMalloc(&s_d_spore_cells, s_cap_spores * sizeof(glm::ivec3));
    }

    size_t need_tips = std::max<size_t>(io_tips.size() * 2 + 256, 4096);
    if (need_tips > s_cap_tips) {
        if (s_d_tips) cudaFree(s_d_tips);
        if (s_d_next_tips) cudaFree(s_d_next_tips);
        if (s_d_proposals) cudaFree(s_d_proposals);
        s_cap_tips = need_tips * 2;
        cudaMalloc(&s_d_tips, s_cap_tips * sizeof(CudaVolTip));
        cudaMalloc(&s_d_next_tips, s_cap_tips * sizeof(CudaVolTip));
        cudaMalloc(&s_d_proposals, s_cap_tips * sizeof(CudaTipProposal));
    }

    const int num_sec_initial = static_cast<int>(io_sections.size());
    cudaMemset(s_d_body_head, 0xFF, body_hash_size * sizeof(int));
    cudaMemcpy(s_d_sections, io_sections.data(), num_sec_initial * sizeof(CudaVolSection), cudaMemcpyHostToDevice);

    // Launch Kernel 1: Spatially hash all existing 3D root sections on GPU
    {
        int threads = 256;
        int blocks = (num_sec_initial + threads - 1) / threads;
        gpu_insert_sections_hash_kernel<<<blocks, threads>>>(
            s_d_sections, num_sec_initial, inv_body_cell, body_hash_mask,
            s_d_body_head, s_d_body_next, s_d_body_cells
        );
    }

    // Launch Kernel 2: Prune occupied nutrient spores and compact surviving spores on GPU
    cudaMemset(s_d_spore_head, 0xFF, spore_hash_size * sizeof(int));
    const int num_spores = static_cast<int>(io_spores.size());
    if (num_spores > 0) {
        cudaMemcpy(s_d_spores, io_spores.data(), num_spores * sizeof(glm::vec3), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (num_spores + threads - 1) / threads;
        gpu_prune_and_hash_spores_kernel<<<blocks, threads>>>(
            s_d_spores, num_spores,
            s_d_sections, s_d_body_head, s_d_body_next, s_d_body_cells,
            body_hash_mask, inv_body_cell, params.kill_radius,
            inv_spore_cell, spore_hash_mask,
            s_d_spore_alive, s_d_spore_head, s_d_spore_next, s_d_spore_cells
        );

        cudaMemset(&s_d_step_counters[0], 0, sizeof(int));
        gpu_compact_surviving_spores_vol_kernel<<<blocks, threads>>>(
            s_d_spores, s_d_spore_alive, num_spores,
            s_d_spores_compact, &s_d_step_counters[0]
        );
        int h_surviving = 0;
        cudaMemcpy(&h_surviving, &s_d_step_counters[0], sizeof(int), cudaMemcpyDeviceToHost);
        h_surviving = std::min(h_surviving, num_spores);
        io_spores.resize(h_surviving);
        if (h_surviving > 0) {
            cudaMemcpy(io_spores.data(), s_d_spores_compact, h_surviving * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        }
    }

    // Launch Kernel 3: Developmental FSPM tip advance — parallel 11-feeler cone
    // sensing, collision handling and lateral branch probing on GPU
    const int num_tips = static_cast<int>(io_tips.size());
    cudaMemcpy(s_d_tips, io_tips.data(), num_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
    {
        int threads = 128;
        int blocks = (num_tips + threads - 1) / threads;
            gpu_dev3d_advance_tips_kernel<<<blocks, threads>>>(
                s_d_tips, num_tips,
                s_d_sections,
                s_d_body_head, s_d_body_next, s_d_body_cells,
                body_hash_mask, inv_body_cell,
                s_d_spores, s_d_spore_head, s_d_spore_next, s_d_spore_cells,
                spore_hash_mask, inv_spore_cell, num_spores,
                s_d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims,
                params, collar_hops, min_skin_gap,
                s_d_proposals
            );

        // Commit non-overlapping new root sections and lateral tips in parallel on GPU
        // (developmental commit kernel, steps_grown bookkeeping).
        int h_init_counters[3] = { num_sec_initial, 0, io_next_root_id };
        cudaMemcpy(&s_d_step_counters[1], h_init_counters, 3 * sizeof(int), cudaMemcpyHostToDevice);

            gpu_dev_commit_proposals_kernel<<<blocks, threads>>>(
                s_d_tips, s_d_proposals, num_tips,
                s_d_sections, &s_d_step_counters[1], static_cast<int>(s_cap_sections),
                s_d_next_tips, &s_d_step_counters[2], static_cast<int>(s_cap_tips),
                &s_d_step_counters[3]
            );
    }

    int h_out_counters[3] = { num_sec_initial, 0, io_next_root_id };
    cudaMemcpy(h_out_counters, &s_d_step_counters[1], 3 * sizeof(int), cudaMemcpyDeviceToHost);
    const int num_sec_after = std::min(h_out_counters[0], static_cast<int>(s_cap_sections));
    const int num_tips_after = std::min(h_out_counters[1], static_cast<int>(s_cap_tips));
    io_next_root_id = h_out_counters[2];

    std::swap(s_d_tips, s_d_next_tips);

    io_sections.resize(num_sec_after);
    if (num_sec_after > 0) {
        cudaMemcpy(io_sections.data(), s_d_sections, num_sec_after * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
    }
    io_tips.resize(num_tips_after);
    if (num_tips_after > 0) {
        cudaMemcpy(io_tips.data(), s_d_tips, num_tips_after * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
    }

    return true;
#else
    (void)vessel_grid;
    (void)params;
    (void)io_spores;
    (void)io_sections;
    (void)io_tips;
    (void)io_next_root_id;
    return false;
#endif
}

bool CudaRasterizer::init_volumetric_seeds_gpu(
    const CudaVolumeGrid& vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    float base_radius,
    float min_vessel_sdf,
    uint32_t rng_seed,
    std::vector<CudaVolSection>& out_sections,
    std::vector<CudaVolTip>& out_tips,
    int& out_next_root_id,
    float wall_clearance_scale,
    int primary_roots
) {
    out_sections.clear();
    out_tips.clear();
    if (!vessel_grid.is_valid() || seed_positions.empty()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int pr = std::clamp(primary_roots, 1, 16);
        int num_seeds = static_cast<int>(seed_positions.size());
        int num_tips = num_seeds * pr;

        glm::vec3* d_seeds = nullptr;
        float* d_vessel_sdf = nullptr;
        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_vessel_sdf, vessel_grid.data.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_secs, num_seeds * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, num_tips * sizeof(CudaVolTip)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_positions.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_vessel_sdf, vessel_grid.data.data(), vessel_grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);

            int threads = 64;
            int blocks = (num_seeds + threads - 1) / threads;
            gpu_init_volumetric_seeds_kernel<<<blocks, threads>>>(
                d_seeds, num_seeds, d_vessel_sdf,
                vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                min_vessel_sdf, base_radius, rng_seed,
                wall_clearance_scale,
                d_secs, d_tips,
                pr
            );

            out_sections.resize(num_seeds);
            out_tips.resize(num_tips);
            cudaMemcpy(out_sections.data(), d_secs, num_seeds * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_tips.data(), d_tips, num_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            out_next_root_id = num_seeds + num_tips;

            cudaFree(d_seeds);
            cudaFree(d_vessel_sdf);
            cudaFree(d_secs);
            cudaFree(d_tips);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_vessel_sdf) cudaFree(d_vessel_sdf);
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
    }
#endif
    return false;
}

bool CudaRasterizer::convert_vol_sections_to_nodes_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    void* out_nodes_data
) {
    if (!sections_data || !out_nodes_data || num_sections == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_sections);
        CudaVolSection* d_secs = nullptr;
        GpuColonizationNodeLayout* d_nodes = nullptr;
        if (cudaMalloc(&d_secs, n * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_nodes, n * sizeof(GpuColonizationNodeLayout)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            gpu_convert_vol_sections_to_nodes_kernel<<<blocks, threads>>>(d_secs, n, d_nodes);
            cudaMemcpy(out_nodes_data, d_nodes, n * sizeof(GpuColonizationNodeLayout), cudaMemcpyDeviceToHost);

            cudaFree(d_secs);
            cudaFree(d_nodes);
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_nodes) cudaFree(d_nodes);
    }
#endif
    return false;
}

bool CudaRasterizer::check_any_awake_tips_gpu(
    const CudaVolTip* tips_data,
    size_t num_tips,
    bool& out_any_awake
) {
    out_any_awake = false;
    if (!tips_data || num_tips == 0) return true;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n = static_cast<int>(num_tips);
        CudaVolTip* d_tips = nullptr;
        int* d_awake = nullptr;
        if (cudaMalloc(&d_tips, n * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_awake, sizeof(int)) == cudaSuccess) {
            cudaMemcpy(d_tips, tips_data, n * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            cudaMemset(d_awake, 0, sizeof(int));
            int threads = 128;
            int blocks = (n + threads - 1) / threads;
            gpu_check_awake_tips_kernel<<<blocks, threads>>>(d_tips, n, d_awake);
            int h_awake = 0;
            cudaMemcpy(&h_awake, d_awake, sizeof(int), cudaMemcpyDeviceToHost);
            out_any_awake = (h_awake > 0);
            cudaFree(d_tips);
            cudaFree(d_awake);
            return true;
        }
        if (d_tips) cudaFree(d_tips);
        if (d_awake) cudaFree(d_awake);
    }
#endif
    return false;
}

bool CudaRasterizer::wake_or_respawn_volumetric_tips_gpu(
    const CudaVolSection* sections_data,
    size_t num_sections,
    std::vector<CudaVolTip>& io_tips,
    bool force_reset_stalled,
    bool& out_any_awake
) {
    out_any_awake = false;
    if (!sections_data || num_sections == 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n_sec = static_cast<int>(num_sections);
        int n_tips = static_cast<int>(io_tips.size());
        int max_tips = std::max(n_tips + 48, 64);

        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;
        int* d_has_child = nullptr;
        int* d_cnt = nullptr; // [0]=awake_count, [1]=tip_count

        if (cudaMalloc(&d_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, max_tips * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_has_child, n_sec * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 2 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_secs, sections_data, n_sec * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            if (n_tips > 0) {
                cudaMemcpy(d_tips, io_tips.data(), n_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            }
            int init_cnt[2] = { 0, n_tips };
            cudaMemcpy(d_cnt, init_cnt, 2 * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            if (n_tips > 0) {
                int t_blocks = (n_tips + threads - 1) / threads;
                if (force_reset_stalled) {
                    gpu_reset_stalled_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips);
                } else {
                    gpu_check_awake_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips, &d_cnt[0]);
                    int h_awake = 0;
                    cudaMemcpy(&h_awake, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost);
                    if (h_awake == 0) {
                        gpu_reset_stalled_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips);
                    }
                }
            }

            bool need_respawn = false;
            if (force_reset_stalled) {
                int h_awake = 0;
                if (n_tips > 0) {
                    cudaMemset(&d_cnt[0], 0, sizeof(int));
                    int t_blocks = (n_tips + threads - 1) / threads;
                    gpu_check_awake_tips_kernel<<<t_blocks, threads>>>(d_tips, n_tips, &d_cnt[0]);
                    cudaMemcpy(&h_awake, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost);
                }
                need_respawn = (h_awake == 0 || n_tips < 6) && (n_sec > 1);
            } else {
                need_respawn = (n_tips == 0 && n_sec > 1);
            }

            if (need_respawn) {
                cudaMemset(d_has_child, 0, n_sec * sizeof(int));
                int s_blocks = (n_sec + threads - 1) / threads;
                gpu_mark_section_children_kernel<<<s_blocks, threads>>>(d_secs, n_sec, d_has_child);
                if (n_tips > 0) {
                    int t_blocks = (n_tips + threads - 1) / threads;
                    gpu_mark_active_tip_sections_kernel<<<t_blocks, threads>>>(d_tips, n_tips, n_sec, d_has_child);
                }
                gpu_respawn_leaf_tips_kernel<<<s_blocks, threads>>>(
                    d_secs, d_has_child, n_sec, d_tips, &d_cnt[1], max_tips
                );
            }

            int h_final_tips = n_tips;
            cudaMemcpy(&h_final_tips, &d_cnt[1], sizeof(int), cudaMemcpyDeviceToHost);
            h_final_tips = std::min(h_final_tips, max_tips);

            io_tips.resize(h_final_tips);
            if (h_final_tips > 0) {
                cudaMemcpy(io_tips.data(), d_tips, h_final_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
                cudaMemset(&d_cnt[0], 0, sizeof(int));
                int t_blocks = (h_final_tips + threads - 1) / threads;
                gpu_check_awake_tips_kernel<<<t_blocks, threads>>>(d_tips, h_final_tips, &d_cnt[0]);
                int h_awake = 0;
                cudaMemcpy(&h_awake, &d_cnt[0], sizeof(int), cudaMemcpyDeviceToHost);
                out_any_awake = (h_awake > 0);
            }

            cudaFree(d_secs);
            cudaFree(d_tips);
            cudaFree(d_has_child);
            cudaFree(d_cnt);
            return true;
        }
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
        if (d_has_child) cudaFree(d_has_child);
        if (d_cnt) cudaFree(d_cnt);
    }
#endif
    return false;
}

bool CudaRasterizer::sync_volumetric_seed_freeze_and_spawn_gpu(
    const CudaVolumeGrid& vessel_grid,
    const std::vector<glm::vec3>& seed_positions,
    const std::vector<uint8_t>& frozen_seeds,
    float base_radius,
    float min_vessel_sdf,
    uint32_t rng_seed,
    float wall_clearance_scale,
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int& io_next_root_id,
    bool& out_spawned_new,
    int primary_roots
) {
    out_spawned_new = false;
    if (!vessel_grid.is_valid() || seed_positions.empty()) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int pr = std::clamp(primary_roots, 1, 16);
        int num_seeds = static_cast<int>(seed_positions.size());
        int num_secs = static_cast<int>(io_sections.size());
        int num_tips = static_cast<int>(io_tips.size());
        int max_secs = num_secs + num_seeds + 16;
        int max_tips = num_tips + num_seeds * pr + 32;

        std::vector<uint8_t> h_frozen(num_seeds, 0);
        for (int s = 0; s < num_seeds && s < static_cast<int>(frozen_seeds.size()); ++s) {
            h_frozen[s] = frozen_seeds[s];
        }

        glm::vec3* d_seeds = nullptr;
        uint8_t* d_frozen = nullptr;
        int* d_seed_exists = nullptr;
        float* d_vessel_sdf = nullptr;
        CudaVolSection* d_secs = nullptr;
        CudaVolTip* d_tips = nullptr;
        int* d_cnt = nullptr; // [0]=num_secs, [1]=num_tips, [2]=next_root_id, [3]=spawned_count

        if (cudaMalloc(&d_seeds, num_seeds * sizeof(glm::vec3)) == cudaSuccess &&
            cudaMalloc(&d_frozen, num_seeds * sizeof(uint8_t)) == cudaSuccess &&
            cudaMalloc(&d_seed_exists, num_seeds * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_vessel_sdf, vessel_grid.data.size() * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_secs, max_secs * sizeof(CudaVolSection)) == cudaSuccess &&
            cudaMalloc(&d_tips, max_tips * sizeof(CudaVolTip)) == cudaSuccess &&
            cudaMalloc(&d_cnt, 4 * sizeof(int)) == cudaSuccess) {

            cudaMemcpy(d_seeds, seed_positions.data(), num_seeds * sizeof(glm::vec3), cudaMemcpyHostToDevice);
            cudaMemcpy(d_frozen, h_frozen.data(), num_seeds * sizeof(uint8_t), cudaMemcpyHostToDevice);
            cudaMemset(d_seed_exists, 0, num_seeds * sizeof(int));
            cudaMemcpy(d_vessel_sdf, vessel_grid.data.data(), vessel_grid.data.size() * sizeof(float), cudaMemcpyHostToDevice);

            if (num_secs > 0) {
                cudaMemcpy(d_secs, io_sections.data(), num_secs * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            }
            if (num_tips > 0) {
                cudaMemcpy(d_tips, io_tips.data(), num_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
            }

            int init_cnt[4] = { num_secs, num_tips, std::max(io_next_root_id, num_seeds * 16 + 1), 0 };
            cudaMemcpy(d_cnt, init_cnt, 4 * sizeof(int), cudaMemcpyHostToDevice);

            int threads = 256;
            if (num_secs > 0) {
                int s_blocks = (num_secs + threads - 1) / threads;
                gpu_sync_volumetric_freeze_sections_kernel<<<s_blocks, threads>>>(
                    d_secs, num_secs, d_seeds, d_frozen, num_seeds, d_seed_exists
                );
            }
            if (num_tips > 0) {
                int t_blocks = (num_tips + threads - 1) / threads;
                gpu_sync_volumetric_freeze_tips_kernel<<<t_blocks, threads>>>(
                    d_tips, num_tips, d_secs, num_secs, d_frozen, num_seeds
                );
            }

            int seed_blocks = (num_seeds + 63) / 64;
            gpu_spawn_missing_volumetric_seeds_kernel<<<seed_blocks, 64>>>(
                d_seeds, d_frozen, d_seed_exists, num_seeds,
                d_vessel_sdf, vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                min_vessel_sdf, base_radius, rng_seed, wall_clearance_scale,
                d_secs, &d_cnt[0], max_secs,
                d_tips, &d_cnt[1], max_tips,
                &d_cnt[2], &d_cnt[3],
                pr
            );

            int out_cnt[4] = { num_secs, num_tips, io_next_root_id, 0 };
            cudaMemcpy(out_cnt, d_cnt, 4 * sizeof(int), cudaMemcpyDeviceToHost);

            int final_secs = std::min(out_cnt[0], max_secs);
            int final_tips = std::min(out_cnt[1], max_tips);
            io_next_root_id = out_cnt[2];
            out_spawned_new = (out_cnt[3] > 0);

            io_sections.resize(final_secs);
            if (final_secs > 0) {
                cudaMemcpy(io_sections.data(), d_secs, final_secs * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            }
            io_tips.resize(final_tips);
            if (final_tips > 0) {
                cudaMemcpy(io_tips.data(), d_tips, final_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_seeds);
            cudaFree(d_frozen);
            cudaFree(d_seed_exists);
            cudaFree(d_vessel_sdf);
            cudaFree(d_secs);
            cudaFree(d_tips);
            cudaFree(d_cnt);
            return true;
        }
        if (d_seeds) cudaFree(d_seeds);
        if (d_frozen) cudaFree(d_frozen);
        if (d_seed_exists) cudaFree(d_seed_exists);
        if (d_vessel_sdf) cudaFree(d_vessel_sdf);
        if (d_secs) cudaFree(d_secs);
        if (d_tips) cudaFree(d_tips);
        if (d_cnt) cudaFree(d_cnt);
    }
#endif
    return false;
}

#ifdef __CUDACC__

#endif


#ifdef __CUDACC__
__global__ void gpu_mark_volumetric_sections_for_seed_kernel(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int deleted_seed_idx,
    int* __restrict__ out_keep
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;

    int seed_slot = dev_find_vol_section_seed_slot(sections, num_sections, i);
    out_keep[i] = (seed_slot == deleted_seed_idx) ? 0 : 1;
}

__global__ void gpu_block_prefix_scan_sections_kernel(
    const int* __restrict__ in_flags,
    int n,
    int* __restrict__ out_local_scanned,
    int* __restrict__ out_block_sums
) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    s_data[tid] = (idx < n) ? in_flags[idx] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }

    if (idx < n) {
        out_local_scanned[idx] = s_data[tid];
    }
    if (tid == blockDim.x - 1) {
        out_block_sums[blockIdx.x] = s_data[tid];
    }
}

__global__ void gpu_single_block_scan_sections_kernel(
    int* __restrict__ io_block_sums,
    int num_blocks
) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    s_data[tid] = (tid < num_blocks) ? io_block_sums[tid] : 0;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int temp = 0;
        if (tid >= offset) {
            temp = s_data[tid - offset];
        }
        __syncthreads();
        s_data[tid] += temp;
        __syncthreads();
    }
    if (tid < num_blocks) {
        io_block_sums[tid] = s_data[tid];
    }
}

__global__ void gpu_add_scanned_block_offsets_sections_kernel(
    const int* __restrict__ local_scanned,
    const int* __restrict__ block_sums,
    const int* __restrict__ in_flags,
    int n,
    int* __restrict__ out_old_to_new_map
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (in_flags[idx] == 0) {
        out_old_to_new_map[idx] = -1;
    } else {
        int block_offset = (blockIdx.x > 0) ? block_sums[blockIdx.x - 1] : 0;
        out_old_to_new_map[idx] = (local_scanned[idx] - 1) + block_offset;
    }
}

__global__ void gpu_remap_and_compact_vol_sections_kernel(
    const CudaVolSection* __restrict__ in_sections,
    int num_sections,
    const int* __restrict__ old_to_new_map,
    int deleted_seed_idx,
    CudaVolSection* __restrict__ out_sections
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sections) return;
    int new_i = old_to_new_map[i];
    if (new_i < 0) return;

    CudaVolSection s = in_sections[i];
    if (s.parent_sec >= 0 && s.parent_sec < num_sections) {
        s.parent_sec = old_to_new_map[s.parent_sec];
    } else {
        s.parent_sec = -1;
    }

    if (s.parent_sec < 0) {
        if (s.root_id > deleted_seed_idx) {
            s.root_id -= 1;
        }
    }

    out_sections[new_i] = s;
}

__global__ void gpu_remap_and_compact_vol_tips_kernel(
    const CudaVolTip* __restrict__ in_tips,
    int num_tips,
    const int* __restrict__ old_to_new_map,
    int num_sections,
    CudaVolTip* __restrict__ out_tips,
    int* __restrict__ d_out_tip_count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_tips) return;

    CudaVolTip tip = in_tips[i];
    int old_sec = tip.sec_idx;
    if (old_sec < 0 || old_sec >= num_sections) return;

    int new_sec = old_to_new_map[old_sec];
    if (new_sec >= 0) {
        tip.sec_idx = new_sec;
        int slot = atomicAdd(d_out_tip_count, 1);
        out_tips[slot] = tip;
    }
}
#endif

bool CudaRasterizer::prune_volumetric_sections_by_seed_gpu(
    std::vector<CudaVolSection>& io_sections,
    std::vector<CudaVolTip>& io_tips,
    int deleted_seed_idx
) {
    if (io_sections.empty() || deleted_seed_idx < 0) return false;
#ifdef __CUDACC__
    if (is_cuda_available()) {
        int n_sec = static_cast<int>(io_sections.size());
        int n_tips = static_cast<int>(io_tips.size());

        CudaVolSection* d_in_secs = nullptr;
        CudaVolSection* d_out_secs = nullptr;
        int* d_keep = nullptr;
        int* d_local_scanned = nullptr;
        int* d_block_sums = nullptr;
        int* d_old_to_new = nullptr;
        CudaVolTip* d_in_tips = nullptr;
        CudaVolTip* d_out_tips = nullptr;
        int* d_tip_count = nullptr;

        const int threads = 256;
        const int blocks = (n_sec + threads - 1) / threads;

        bool ok = (cudaMalloc(&d_in_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess) &&
                  (cudaMalloc(&d_out_secs, n_sec * sizeof(CudaVolSection)) == cudaSuccess) &&
                  (cudaMalloc(&d_keep, n_sec * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_local_scanned, n_sec * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_block_sums, blocks * sizeof(int)) == cudaSuccess) &&
                  (cudaMalloc(&d_old_to_new, n_sec * sizeof(int)) == cudaSuccess);
        if (ok && n_tips > 0) {
            ok = (cudaMalloc(&d_in_tips, n_tips * sizeof(CudaVolTip)) == cudaSuccess) &&
                 (cudaMalloc(&d_out_tips, n_tips * sizeof(CudaVolTip)) == cudaSuccess) &&
                 (cudaMalloc(&d_tip_count, sizeof(int)) == cudaSuccess);
        }

        if (ok) {
            cudaMemcpy(d_in_secs, io_sections.data(), n_sec * sizeof(CudaVolSection), cudaMemcpyHostToDevice);
            if (n_tips > 0) {
                cudaMemcpy(d_in_tips, io_tips.data(), n_tips * sizeof(CudaVolTip), cudaMemcpyHostToDevice);
                cudaMemset(d_tip_count, 0, sizeof(int));
            }

            // 1. Mark keep (1) or discard (0) based on origin seed
            gpu_mark_volumetric_sections_for_seed_kernel<<<blocks, threads>>>(
                d_in_secs, n_sec, deleted_seed_idx, d_keep
            );

            // 2. Multi-block prefix scan
            size_t shared_bytes = threads * sizeof(int);
            gpu_block_prefix_scan_sections_kernel<<<blocks, threads, shared_bytes>>>(
                d_keep, n_sec, d_local_scanned, d_block_sums
            );
            if (blocks > 1) {
                gpu_single_block_scan_sections_kernel<<<1, 1024, 1024 * sizeof(int)>>>(
                    d_block_sums, blocks
                );
            }
            gpu_add_scanned_block_offsets_sections_kernel<<<blocks, threads>>>(
                d_local_scanned, d_block_sums, d_keep, n_sec, d_old_to_new
            );

            // 3. Compact and remap sections
            gpu_remap_and_compact_vol_sections_kernel<<<blocks, threads>>>(
                d_in_secs, n_sec, d_old_to_new, deleted_seed_idx, d_out_secs
            );

            int h_surviving_secs = 0;
            if (blocks > 0) {
                cudaMemcpy(&h_surviving_secs, &d_block_sums[blocks - 1], sizeof(int), cudaMemcpyDeviceToHost);
            }
            h_surviving_secs = std::clamp(h_surviving_secs, 0, n_sec);

            // 4. Compact and remap active tips
            int h_surviving_tips = 0;
            if (n_tips > 0) {
                int tip_blocks = (n_tips + threads - 1) / threads;
                gpu_remap_and_compact_vol_tips_kernel<<<tip_blocks, threads>>>(
                    d_in_tips, n_tips, d_old_to_new, n_sec, d_out_tips, d_tip_count
                );
                cudaMemcpy(&h_surviving_tips, d_tip_count, sizeof(int), cudaMemcpyDeviceToHost);
                h_surviving_tips = std::clamp(h_surviving_tips, 0, n_tips);
            }

            // 5. Copy back to host
            io_sections.resize(h_surviving_secs);
            if (h_surviving_secs > 0) {
                cudaMemcpy(io_sections.data(), d_out_secs, h_surviving_secs * sizeof(CudaVolSection), cudaMemcpyDeviceToHost);
            }

            io_tips.resize(h_surviving_tips);
            if (h_surviving_tips > 0) {
                cudaMemcpy(io_tips.data(), d_out_tips, h_surviving_tips * sizeof(CudaVolTip), cudaMemcpyDeviceToHost);
            }

            cudaFree(d_in_secs);
            cudaFree(d_out_secs);
            cudaFree(d_keep);
            cudaFree(d_local_scanned);
            cudaFree(d_block_sums);
            cudaFree(d_old_to_new);
            if (d_in_tips) cudaFree(d_in_tips);
            if (d_out_tips) cudaFree(d_out_tips);
            if (d_tip_count) cudaFree(d_tip_count);
            return true;
        }

        if (d_in_secs) cudaFree(d_in_secs);
        if (d_out_secs) cudaFree(d_out_secs);
        if (d_keep) cudaFree(d_keep);
        if (d_local_scanned) cudaFree(d_local_scanned);
        if (d_block_sums) cudaFree(d_block_sums);
        if (d_old_to_new) cudaFree(d_old_to_new);
        if (d_in_tips) cudaFree(d_in_tips);
        if (d_out_tips) cudaFree(d_out_tips);
        if (d_tip_count) cudaFree(d_tip_count);
    }
#endif
    return false;
}

#ifdef __CUDACC__


#endif // __CUDACC__


} // namespace rhizome

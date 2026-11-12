#include "engine/sdf/cuda_rasterizer.hpp"
#include "engine/sdf/cuda_common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

namespace rhizome {

#ifdef __CUDACC__

// --- 2. CUDA Kernel: Spore Search ---
__global__ void spore_search_kernel(
    CudaSporeData* __restrict__ spores,
    int num_spores,
    const CudaBranchNode* __restrict__ nodes,
    int num_nodes,
    float search_radius_sq,
    float kill_radius_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_spores) return;

    CudaSporeData sp = spores[idx];
    if (!sp.alive) return;

    float closest_dist_sq = search_radius_sq;
    int closest_idx = -1;
    bool kill = false;

    for (int n = 0; n < num_nodes; ++n) {
        CudaBranchNode node = nodes[n];
        glm::vec3 diff = sp.pos - node.pos;
        float d2 = glm::dot(diff, diff);

        float eff_kill_r = node.radius + sqrtf(kill_radius_sq);
        if (d2 <= eff_kill_r * eff_kill_r) {
            kill = true;
            break;
        }

        if (d2 < closest_dist_sq) {
            closest_dist_sq = d2;
            closest_idx = n;
        }
    }

    if (kill) {
        spores[idx].alive = false;
        spores[idx].closest_node_idx = -1;
        spores[idx].min_dist = 0.0f;
    } else {
        spores[idx].closest_node_idx = closest_idx;
        spores[idx].min_dist = sqrtf(closest_dist_sq);
    }
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// float atomicAdd accumulation replaced by fixed-point uint64 atomicAdd (Q = 2^24,
// quantum ~6e-8). CUDA exposes no signed 64-bit atomicAdd overload, so the shadow
// accumulators are unsigned long long: two's-complement addition mod 2^64 is
// associative and commutative, so the accumulated total is bit-identical
// regardless of thread scheduling (deterministic). The original float buffer is
// refilled by a dequantize pass for existing consumers.
// Track B fix 1 (2026-10-22): original draft used long long, which does not
// compile (no such atomicAdd overload) — flavor changed to uint64, semantics equal.
__global__ void gpu_accumulate_spore_forces_kernel(
    const CudaSporeData* __restrict__ spores,
    int num_spores,
    const CudaBranchNode* __restrict__ nodes,
    int num_nodes,
    unsigned long long* __restrict__ out_forces_q,
    int* __restrict__ out_counts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_spores) return;

    CudaSporeData sp = spores[i];
    if (sp.alive && sp.closest_node_idx >= 0 && sp.closest_node_idx < num_nodes) {
        int ni = sp.closest_node_idx;
        glm::vec3 d = sp.pos - nodes[ni].pos;
        float len2 = glm::dot(d, d);
        if (len2 > 1e-12f) {
            glm::vec3 unit_d = d * rsqrtf(len2);
            atomicAdd(&out_forces_q[ni * 3 + 0], (unsigned long long)llroundf(unit_d.x * 16777216.0f));
            atomicAdd(&out_forces_q[ni * 3 + 1], (unsigned long long)llroundf(unit_d.y * 16777216.0f));
            atomicAdd(&out_forces_q[ni * 3 + 2], (unsigned long long)llroundf(unit_d.z * 16777216.0f));
            atomicAdd(&out_counts[ni], 1);
        }
    }
}

// v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B:
// dequantize pass for the fixed-point uint64 accumulators (Q = 2^24). One thread
// per element; per-element conversion is deterministic by construction. The
// unsigned->signed conversion is well-defined modulo 2^64 under C++20
// (CMAKE_CUDA_STANDARD 20), so the two's-complement sum maps to the signed total.
__global__ void gpu_dequantize_ll_to_float_kernel(
    const unsigned long long* __restrict__ q,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<float>(static_cast<long long>(q[i])) * (1.0f / 16777216.0f);
}

#endif // __CUDACC__

// 2. Colonization Spore Search
bool CudaRasterizer::evaluate_spores(
    std::vector<CudaSporeData>& spores,
    const std::vector<CudaBranchNode>& nodes,
    float search_radius,
    float kill_radius,
    std::vector<glm::vec3>& out_attraction_forces,
    std::vector<int>& out_attraction_counts,
    float& out_time_ms
) {
    auto start = std::chrono::high_resolution_clock::now();
    out_attraction_forces.assign(nodes.size(), glm::vec3(0.0f));
    out_attraction_counts.assign(nodes.size(), 0);

    float search_r2 = search_radius * search_radius;
    float kill_r2 = kill_radius * kill_radius;

#ifdef __CUDACC__
    if (is_cuda_available() && !spores.empty() && !nodes.empty()) {
        CudaSporeData* d_spores = nullptr;
        CudaBranchNode* d_nodes = nullptr;
        float* d_forces = nullptr;
        int* d_counts = nullptr;
        unsigned long long* d_forces_q = nullptr; // v24.02 P1.2 DRAFT: quantized shadow accumulator (Q = 2^24, uint64)
        size_t num_spores = spores.size();
        size_t num_nodes = nodes.size();

        if (cudaMalloc(&d_spores, num_spores * sizeof(CudaSporeData)) == cudaSuccess &&
            cudaMalloc(&d_nodes, num_nodes * sizeof(CudaBranchNode)) == cudaSuccess &&
            cudaMalloc(&d_forces, num_nodes * 3 * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&d_counts, num_nodes * sizeof(int)) == cudaSuccess &&
            cudaMalloc(&d_forces_q, num_nodes * 3 * sizeof(unsigned long long)) == cudaSuccess) {

            cudaMemcpy(d_spores, spores.data(), num_spores * sizeof(CudaSporeData), cudaMemcpyHostToDevice);
            cudaMemcpy(d_nodes, nodes.data(), num_nodes * sizeof(CudaBranchNode), cudaMemcpyHostToDevice);
            cudaMemset(d_forces, 0, num_nodes * 3 * sizeof(float));
            cudaMemset(d_counts, 0, num_nodes * sizeof(int));
            cudaMemset(d_forces_q, 0, num_nodes * 3 * sizeof(unsigned long long));

            int threads = 256;
            int blocks = (static_cast<int>(num_spores) + threads - 1) / threads;
            spore_search_kernel<<<blocks, threads>>>(
                d_spores, static_cast<int>(num_spores), d_nodes, static_cast<int>(num_nodes), search_r2, kill_r2
            );
            gpu_accumulate_spore_forces_kernel<<<blocks, threads>>>(
                d_spores, static_cast<int>(num_spores), d_nodes, static_cast<int>(num_nodes), d_forces_q, d_counts
            );
            // v24.02 P1.2 DRAFT (2026-10-22) — UNVERIFIED UNTIL TRACK B: dequantize pass
            int dq_blocks = (static_cast<int>(num_nodes * 3) + threads - 1) / threads;
            gpu_dequantize_ll_to_float_kernel<<<dq_blocks, threads>>>(
                d_forces_q, d_forces, static_cast<int>(num_nodes * 3)
            );

            cudaMemcpy(spores.data(), d_spores, num_spores * sizeof(CudaSporeData), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_attraction_forces.data(), d_forces, num_nodes * 3 * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(out_attraction_counts.data(), d_counts, num_nodes * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_spores);
            cudaFree(d_nodes);
            cudaFree(d_forces);
            cudaFree(d_counts);
            cudaFree(d_forces_q);

            auto end = std::chrono::high_resolution_clock::now();
            out_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
            return true;
        }
        if (d_spores) cudaFree(d_spores);
        if (d_nodes) cudaFree(d_nodes);
        if (d_forces) cudaFree(d_forces);
        if (d_counts) cudaFree(d_counts);
        if (d_forces_q) cudaFree(d_forces_q);
    }
#endif

    std::cerr << "[CUDA Error] GPU acceleration required but unavailable or out of memory in evaluate_spores.\n";
    return false;
}

#ifdef __CUDACC__
__global__ void gpu_sample_spores_kernel(
    const float* __restrict__ vessel_sdf,
    glm::ivec3 vessel_min_coord,
    glm::ivec3 vessel_dims,
    float voxel_size,
    float min_x, float min_y, float min_z,
    float span_x, float span_y, float span_z,
    float spore_margin,
    uint32_t seed,
    int attempts_per_thread,
    int target_count,
    glm::vec3* __restrict__ out_spores,
    int* __restrict__ d_valid_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState_t rng;
    curand_init(seed + static_cast<uint32_t>(idx * 1664525u + 1013904223u), idx, 0, &rng);

    for (int step = 0; step < attempts_per_thread; ++step) {
        if (*d_valid_count >= target_count) break;

        float rx = curand_uniform(&rng);
        float ry = curand_uniform(&rng);
        float rz = curand_uniform(&rng);

        glm::vec3 cand(min_x + rx * span_x, min_y + ry * span_y, min_z + rz * span_z);
        float sdf = dev_sample_sdf_trilinear(vessel_sdf, vessel_min_coord, vessel_dims, voxel_size, cand);

        if (sdf <= spore_margin) {
            int slot = atomicAdd(d_valid_count, 1);
            if (slot < target_count) {
                out_spores[slot] = cand;
            } else {
                break;
            }
        }
    }
}
#endif

bool CudaRasterizer::sample_spores_gpu(
    const CudaVolumeGrid& vessel_grid,
    int target_spore_count,
    uint32_t seed,
    float spore_margin,
    std::vector<glm::vec3>& out_spores
) {
    if (!vessel_grid.is_valid() || target_spore_count <= 0) return false;

#ifdef __CUDACC__
    if (is_cuda_available()) {
        float* d_grid = nullptr;
        size_t grid_bytes = vessel_grid.data.size() * sizeof(float);
        if (cudaMalloc(&d_grid, grid_bytes) == cudaSuccess) {
            cudaMemcpy(d_grid, vessel_grid.data.data(), grid_bytes, cudaMemcpyHostToDevice);

            glm::vec3 min_w = glm::vec3(vessel_grid.min_coord) * vessel_grid.voxel_size;
            glm::vec3 max_w = glm::vec3(vessel_grid.min_coord + vessel_grid.dims) * vessel_grid.voxel_size;
            glm::vec3 span = max_w - min_w;

            const int attempts_per_thread = 32;
            const int total_attempts = target_spore_count * 50;
            const int total_threads = std::max(256, (total_attempts + attempts_per_thread - 1) / attempts_per_thread);

            glm::vec3* d_out_spores = nullptr;
            int* d_valid_count = nullptr;

            if (cudaMalloc(&d_out_spores, target_spore_count * sizeof(glm::vec3)) == cudaSuccess &&
                cudaMalloc(&d_valid_count, sizeof(int)) == cudaSuccess) {

                cudaMemset(d_valid_count, 0, sizeof(int));

                int threads = 256;
                int blocks = (total_threads + threads - 1) / threads;
                gpu_sample_spores_kernel<<<blocks, threads>>>(
                    d_grid, vessel_grid.min_coord, vessel_grid.dims, vessel_grid.voxel_size,
                    min_w.x, min_w.y, min_w.z, span.x, span.y, span.z,
                    spore_margin, seed, attempts_per_thread, target_spore_count,
                    d_out_spores, d_valid_count
                );

                int h_count = 0;
                cudaMemcpy(&h_count, d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
                h_count = std::min(h_count, target_spore_count);

                if (h_count > 0) {
                    out_spores.resize(h_count);
                    cudaMemcpy(out_spores.data(), d_out_spores, h_count * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
                }

                cudaFree(d_out_spores);
                cudaFree(d_valid_count);
                cudaFree(d_grid);
                return (h_count > 0);
            }

            if (d_out_spores) cudaFree(d_out_spores);
            if (d_valid_count) cudaFree(d_valid_count);
            cudaFree(d_grid);
        }
    }
#endif

    return !out_spores.empty();
}

#ifdef __CUDACC__

__global__ void gpu_compact_surviving_spores_kernel(
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

#endif

} // namespace rhizome

# Rhizome Native — implicit SDF engine

A native C++20 + CUDA 12 engine for implicit-surface modelling and structure growth.
Every geometric stage runs in CUDA kernels; the CPU streams files and draws the interface.

Project navigation: `CODE_MAP.md` covers the module map, entry points and host/device
boundaries of the `native/src/{core,ui,engine,verification}` tree.

**Status: v24.23** — 256 controls / 26 sections — `--verify` **all tests** on the target
machine (Windows 11, VS2022, CUDA 12.4, RTX 3060 / sm_86). Build and CLI usage live in
the top-level README.

**Fast SDF path.** For grids at or above 32M voxels the mesh-to-SDF build uses an
O(surface + V) pipeline: exact-distance band splat, scanline parity signs, an
exact-winding patch for a small number of mismatched sites, and FSM 
re-distancing of the far field. That path is what makes 0.10 mm the shipped default
Detail Level. Every verification grid sits below the 32M-voxel gate, so `--verify`
results are unaffected by it.

**Known characteristic.** Exports of the growth-affected chain are not bit-reproducible
run-to-run: measured spread at fixed inputs is about ±5% on section and vertex counts.
Compare distributions, not single runs.

---

## 1. Dimensional Tolerance & Cell Size Derivation

### Requirement
* **Maximum Chord Error:** $\epsilon_{\max} \le 0.1\,\text{mm}$
* **Minimum Curvature Radius:** $R_{\min} \ge 0.5\,\text{mm}$

### Mathematical Derivation
For an osculating circle of radius $R$ subtended by a planar polygonal edge (chord) of length $h$, the sagitta (chord error $\epsilon$) is:
$$\epsilon = R - \sqrt{R^2 - \left(\frac{h}{2}\right)^2}$$

Using the second-order Taylor expansion for $h \ll R$:
$$\epsilon \approx \frac{h^2}{8R} \implies h \le \sqrt{8 R \epsilon_{\max}}$$

Substituting $R_{\min} = 0.5\,\text{mm}$ and $\epsilon_{\max} = 0.1\,\text{mm}$:
$$h_{\max} \le \sqrt{8 \times 0.5 \times 0.1} = \sqrt{0.4} \approx 0.6324\,\text{mm}$$

In a 3D dual contouring / marching voxel grid with isotropic cell size $\Delta x$:
* Face diagonal chord length: $h_{\text{face}} = \Delta x \sqrt{2} \approx 1.4142 \Delta x$
* Space diagonal chord length: $h_{\text{space}} = \Delta x \sqrt{3} \approx 1.7321 \Delta x$

To strictly enforce that $h_{\text{face}} \le 0.6324\,\text{mm}$:
$$\Delta x \le \frac{0.6324}{\sqrt{2}} \approx 0.447\,\text{mm}$$

### Cell Sizing Strategy
1. **Nominal Cell Size:** $\mathbf{0.50\,\text{mm}}$ in near-surface narrow-band regions where local curvature $R \ge 1.0\,\text{mm}$.
   * At $R = 1.0\,\text{mm}$, $\Delta x = 0.5\,\text{mm}$:
     $$\epsilon = 1.0 - \sqrt{1.0^2 - (0.3535)^2} \approx 0.065\,\text{mm} < 0.1\,\text{mm} \quad \text{[SATISFIED]}$$
2. **Adaptive Refinement:** $\mathbf{0.25\,\text{mm}}$ in regions where curvature radius $R \in [0.5\,\text{mm}, 1.0\,\text{mm}]$.
   * At $R = 0.5\,\text{mm}$, $\Delta x = 0.25\,\text{mm}$:
     $$\epsilon = 0.5 - \sqrt{0.5^2 - (0.1768)^2} \approx 0.032\,\text{mm} \ll 0.1\,\text{mm} \quad \text{[SATISFIED]}$$

---

## 2. Phase 1 Architecture & Deliverables

### 1. True Euclidean Distance via 7 Voronoi Regions
* Implemented inline in `mesh_to_sdf_kernel` (`native/src/engine/sdf/cuda_rasterizer.cu`) via `point_triangle_distance_sq` (`native/src/core/math_utils.cuh`); the same closest-point routine is reused by the GPU mesh-ops kernels (`native/src/engine/mesh/cuda_mesh_ops.cu`).
* Employs Christer Ericson / David Eberly Voronoi feature decomposition across the 7 spatial regions of a triangle (Face Interior, 3 Edges, 3 Vertices).
* Guaranteed true Euclidean metric ($\|\nabla \phi\| = 1$) without square root shortcuts or planar clamping.
* Accelerated by native CUDA parallelization over voxels with tile-binned triangle headers (`CudaTileHeader`) — the entire distance query runs on the GPU. **

### 2. Robust Sign Determination via Generalized Winding Numbers
* Implemented in `gpu_compute_winding_numbers_kernel` (`native/src/engine/mesh/cuda_mesh_ops.cu`), dispatched as `CudaRasterizer::compute_winding_numbers_gpu()` (`native/src/engine/sdf/cuda_rasterizer.hpp`); the optional normalizer pass is configured by `SdfConfig::winding_number_theta` / `SdfConfig::enable_winding_normalizer` (`native/src/engine/sdf/narrow_band_sdf.hpp`).
* Implements the solid-angle formulation of Jacobson et al. (2013) / Barill et al. (2018): the per-point winding number is the GPU-reduced summed solid angle divided by 4π.
* **Parity raycasting has been removed entirely.**
* Resolves non-manifold edges, open boundaries, zero-thickness shells, and inverted normals cleanly without topological leaks. **

### 3. Sparse Volumetric Narrow-Band Grid (Native CUDA 3D Volume)
* Implemented in `native/src/engine/sdf/narrow_band_sdf.hpp` and `native/src/engine/sdf/narrow_band_sdf.cpp` (headers co-located since v23.29).
* Allocates only active voxels within $\pm 3.0 \times \Delta x$ of the zero crossing ($[-1.5\,\text{mm}, +1.5\,\text{mm}]$).
* Unbounded background distance clamped to external field value.

### 4. Manifold OBJ Exporter
* Implemented in `native/src/engine/io/obj_exporter.hpp` and `native/src/engine/io/obj_exporter.cpp` (headers co-located since v23.29).
* **Scale:** $1.0\,\text{unit} = 1.0\,\text{mm}$.
* **Precision:** ASCII with fixed 6-decimal precision (`0.000000`).
* **Topological Verification:** Automatically audits that every undirected edge has exactly 2 incident faces with opposing cyclic orientations, zero boundary loops (watertight), and zero non-manifold edges before export.

### 5. Verification Harness & Acceptance Criteria
* Implemented in `native/src/verification/harness.hpp` and `native/src/verification/harness.cpp` (headers co-located since v23.29).
* Evaluates extracted isosurface vertices against exact analytical equations:
  * **Sphere:** $F(x,y,z) = \sqrt{x^2 + y^2 + z^2} - 20.0 = 0$
  * **Torus:** $F(x,y,z) = \sqrt{(\sqrt{x^2 + y^2} - 25.0)^2 + z^2} - 5.0 = 0$
  * **Filleted Box:** Rounded distance box with $r = 2.0\,\text{mm}$ fillet.
* **Criterion:** Maximum deviation $\le 0.100\,\text{mm}$ on Sphere and Torus.
* **Scratch files:** the algorithm-file round-trip (Test 19) writes its scratch
  `.json`, `.cu` and `.ini` files to the process working directory, not `%TEMP%`;
  this keeps the test runnable on machines with a locked-down system temp folder.

---

## 3. Verification Harness Benchmark Results

| Primitive | Input Tris | Extracted Verts | Extracted Tris | Voxel Size | Max Deviation | Mean Deviation | RMS Deviation | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sphere ($R = 20\,\text{mm}$)** | 32,768 | 41,218 | 82,432 | $0.50\,\text{mm}$ | **$0.0631\,\text{mm}$** | **$0.0214\,\text{mm}$** | **$0.0258\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |
| **Torus ($R = 25, r = 5\,\text{mm}$)** | 32,768 | 58,490 | 116,976 | $0.40\,\text{mm}$ | **$0.0718\,\text{mm}$** | **$0.0289\,\text{mm}$** | **$0.0341\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |
| **Filleted Box ($r = 2\,\text{mm}$)** | 19,200 | 36,912 | 73,824 | $0.40\,\text{mm}$ | **$0.0842\,\text{mm}$** | **$0.0312\,\text{mm}$** | **$0.0387\,\text{mm}$** | **PASSED** ($\le 0.100\,\text{mm}$) |

*Note on Torus & High Curvature Areas:* On the inner ring of the torus where the Gaussian curvature is negative and principal curvature $k_1 = \frac{1}{5.0\,\text{mm}} = 0.2\,\text{mm}^{-1}$, a uniform $0.50\,\text{mm}$ cell size approaches the chord error upper bound ($\sim 0.089\,\text{mm}$). Refining the cell to $0.40\,\text{mm}$ or $0.25\,\text{mm}$ reduces maximum deviation to $0.0718\,\text{mm}$, safely within the $0.1\,\text{mm}$ acceptance criterion.

---

## 4. Building on Windows 11 Pro

### Prerequisites
1. **Visual Studio 2022** (MSVC v143 toolset with C++20 and Windows 11 SDK).
2. **CUDA Toolkit 12.x** (with Compute Capability 8.6 support).
3. **CMake 3.24+**.
4. **vcpkg** package manager (`VCPKG_ROOT` environment variable configured).

### Build Command
```cmd
cd native
vcpkg install
build.bat
```

Or end-to-end with automatic vcpkg bootstrap and the v23.29 build-time CSV<->registry audit: `powershell -File build.ps1` (the audit runs as the `rhizome_control_audit` CMake target whenever a Python 3 interpreter is available).

Post-build gates:
```cmd
.\build\Release\rhizome_app.exe --verify      :: registry audit + full 19-test GPU suite (headless, exit-code gated)
.\build\Release\rhizome_app.exe --benchmark   :: automated GPU benchmark -> logs\gpu_benchmark_report.{json,md}
```

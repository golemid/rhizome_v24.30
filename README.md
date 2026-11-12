# Rhizome — native SDF growth engine

**Status: v24.23** · 256 controls / 26 sections · `--verify` **all tests PASS** on the target
machine (Windows 11, VS2022, CUDA 12.4, RTX 3060 / sm_86, Dear ImGui 1.92.9b).

A native Windows desktop application that loads a mesh, builds a signed distance
field on the GPU, grows a kelp structure through it, meshes the
result, and exports a watertight object for resin printing.

> v24.23 kelp-reorg: all retired growth engines and their dead code are fully
> removed; the single Kelp engine (mode 6) lives in `native/src/engine/kelp/`.
> See [LEFTOVER_SCAN.md](./LEFTOVER_SCAN.md) for the removal report.

No CPU fallback for any computational stage. Everything geometric runs
in CUDA kernels; the CPU streams files and draws the interface.

---

## Growth algorithm

| Mode | Name | What it is |
| :--- | :--- | :--- |
| 6 | Kelp (Marine FSPM) | 3D volumetric functional-structural plant model: holdfast anchoring, upward buoyancy, wave drift, per-order calibers, SLA print floor |

This build ships **one engine**. The other modes this project once carried (0 Corals,
1 Roots, 2 Vascular, 3 Creeping, 5 Root Typ 3D) have been removed from the codebase,
.

The algorithm file format is unchanged and the interface still supports importing,
exporting, switching and deleting algorithms. Only engine mode 6 is accepted.

---

## Verification (`--verify`)

Headless, exit-code gated. verification tests: analytical geometry, dirty-mesh winding-number
classification, end-to-end vessel pipeline, CUDA unit tests, and serialization
round-trips.

| Primitive | Max deviation (vertex / midpoint) | Result |
| :--- | :--- | :--- |
| Sphere (R=20 mm) | 0.0134 / 0.0139 mm | PASS |
| Torus (R=25, r=5 mm) | 0.0203 / 0.0210 mm | PASS |
| Filleted box (r=2 mm) | 0.0598 / 0.0629 mm | PASS |
| Capsule chain | 0.0883 / 0.0954 mm | PASS |

Acceptance bound: 0.100 mm. CUDA narrow-band rasterisation speedup over the
unaccelerated baseline: ~103–105× (target ≥ 10×; the range is run-to-run variance).

Known characteristic: exports of the growth-affected chain are **not bit-reproducible
run-to-run**. Measured spread at fixed inputs is ±5% on section and vertex counts, so
compare distributions, not single runs — this is what `--repeat` is for.

---

## Build

Requires Visual Studio 2022 (MSVC v143), CUDA Toolkit 12.x, CMake 3.24+, vcpkg.

```powershell
cd native
powershell -File build.ps1        # configures, installs vcpkg deps, audits the
                                  # control registry, builds, runs --verify
```

Build output: `native\build\Release\rhizome_app.exe`.

---

## Run

```powershell
.\build\Release\rhizome_app.exe              # desktop application (or double-click run.bat)
.\build\Release\rhizome_app.exe --verify     # 19-test suite, exit-code gated
.\build\Release\rhizome_app.exe --benchmark  # GPU benchmark runner
```

### Headless growth runs

`--run` drives the growth engine without a window or OpenGL context, so a script or
an agent can run experiments and read the numbers back.

```powershell
rhizome_app.exe --run algo.cu --mesh assets\mug.obj --detail 0.30 --repeat 5 `
                --out result.obj --stats stats.json --dev "dev_grav_gain=0.15"
```

| Flag | Meaning |
| :--- | :--- |
| `--run <file>` | Algorithm profile (`.json` or `.cu`). Required. |
| `--mesh <file>` | Specimen (`.obj` / `.stl`). Omit for a procedural 40 mm box. |
| `--out <file>` | Exported mesh (default `run_out.obj`). |
| `--stats <file>` | JSON metrics (default `run_stats.json`). |
| `--detail <mm>` | SDF voxel size (default 0.20). |
| `--repeat <n>` | Growth passes to average; the SDF is built once (default 1). |
| `--mode <n>` | Override the engine mode; only 6 (Kelp Marine FSPM) is accepted. |
| `--steps <n>` | Override max iterations. |
| `--seed <n>` | Override the growth seed (default 42). |
| `--dev k=v,...` | Developmental overrides, e.g. `dev_noise=0.2,dev_vigor=1.3`. |

The headless path drives the same single engine as the GUI: Kelp Marine FSPM
(mode 6).

Each run also writes `logs/structure_dump_mode<M>_<timestamp>.txt`: one row per
section with position, radius, parent, branch order and root id. That file carries
the topology the mesh cannot, and `scripts/eye/analyze_dump.py` turns it into
branch-order, depth, radius, printability and crowding metrics.

---

## Architecture

`native/src/{core,ui,engine,verification}` with co-located headers.

- **`rhizome_core`** — SDF, the Kelp growth engine (`engine/kelp/`), meshing, I/O, verification. No UI, no OpenGL.
- **`rhizome_app`** — Dear ImGui panels, GL viewport, CLI entry points.

Every live interactive widget is a registered control: `param_controls_complete_list.csv`
is the single source of truth for the catalog, checked against the C++ registry at
build time and again by `--verify`.

---

## Documentation

- [CODE_MAP.md](./CODE_MAP.md) — module map, entry points, host/device boundaries
- [AGENTS.md](./AGENTS.md) — the binding rules for anyone (or anything) changing this tree
- [native/README.md](./native/README.md) — engine and SDF details
- [param_controls_complete_list.csv](./param_controls_complete_list.csv) — the control catalog (256 rows)
- [param_controls_inventory.md](./param_controls_inventory.md) — per-control architectural notes
- [ui_interaction_monitor_system.md](./ui_interaction_monitor_system.md) — telemetry specification
- [scripts/eye/analyze_dump.py](./scripts/eye/analyze_dump.py) — structure-dump analyser

---

## Protected areas

1. `/native` must never be deleted or renamed.
2. GPU-only rule: no CPU fallback for computational work.
3. 4. The existing growth engine is preserved when new ones are added.

# Rhizome UI Parameter Controls — inventory
## Per-control architectural notes

> **Status:** the live catalog is `param_controls_complete_list.csv` — **256 controls /
> 26 sections**, `CTRL001`—`CTRL256`, audited against the C++ registry at build time and
> again by `--verify`. Where this document and the CSV disagree, the CSV wins.

---

## 📊 SUMMARY COUNTS

| Control Type | Rendering Function / API | Instance Count | Source Files |
|:---|:---|:---:|:---|
| **Compact Sliders (Float)** | `render_compact_slider_float` | **99** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Compact Sliders (Int)** | `render_compact_slider_int` | **8** | `panel_growth.cpp`, `panel_scene.cpp` |
| **Toggle Switches** | `render_toggle_switch` | **25** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Action Buttons** | `ImGui::Button` | **34** | all four panel `.cpp` files |
| **Pill Buttons** | `render_pill_button` | **4** | `panel_growth.cpp` |
| **Dropdown Combo Boxes** | `begin_pill_combo`, `render_pill_combo` | **6** | `panel_growth.cpp`, `panel_styles.cpp`, `panel_scene.cpp` |
| **Color Pickers** | `ImGui::ColorEdit4` | **78** | `panel_styles.cpp` |
| **Text Inputs** | `ImGui::InputText` | **1** | `panel_styles.cpp` |
| **Selectable Items** | `ImGui::Selectable` | **1** | `panel_verification.cpp` |
| **TOTAL CATALOG CONTROLS** | | **256** | Complete Codebase Coverage (225 live + 31 retired) |

> **Retired entries (v24.23):** thirty-one rows are marked `RETIRED-01` … `RETIRED-31`
> and are no longer rendered:
>
> - `CTRL009`–`CTRL016` — `RETIRED-01` … `RETIRED-08`, physical drivers of the removed growth modes
> - `CTRL026` — `RETIRED-09`, former steering cone (classic feeler sensor)
> - `CTRL048` — `RETIRED-16`, former Snap to Root
> - `CTRL091`–`CTRL092` — `RETIRED-28` / `RETIRED-29`, former 3D relief-shading parameters
> - `CTRL105`–`CTRL106` — `RETIRED-30` / `RETIRED-31`, former matte-glass sliders
>
> They stay in the catalog and the registry so numbering remains stable. Reuse
> those ids if a future algorithm needs the same parameters.
*Note: All 256 controls are sequentially indexed from `CTRL001` through `CTRL256` in `param_controls_complete_list.csv`. The counts below include the 31 retired catalog rows; 225 controls are live.*

The counts above are the machine-audited census: enforced at build time by the `rhizome_control_audit` CMake target and at runtime by `--verify`. The CSV defines 26 panel sections; the functional groupings below merge them for readability.



---

## 📁 SOURCE FILE BREAKDOWN

### 1. `native/src/ui/panels/panel_growth.cpp` (67 controls)
Contains the growth command bar, algorithm selector, colonization seeds, branching / vessel physics, seeding management, growth execution, and the Kelp Developmental block.

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Growth Command Bar | `CTRL001` – `CTRL004` | 0 | 0 | 4 | 0 | **4** |
| Algorithm | `CTRL005` – `CTRL016` | 8 | 0 | 3 | 1 | **12** |
| Colonization Seeds | `CTRL017` – `CTRL020` | 4 | 0 | 0 | 0 | **4** |
| Branching & Growth | `CTRL021` – `CTRL027` | 6 | 1 | 0 | 0 | **7** |
| Vessel & Caliber | `CTRL028` – `CTRL034` | 6 | 1 | 0 | 0 | **7** |
| Retired catalog rows (CTRL035–CTRL040) | `CTRL035` – `CTRL040` | 6 | 0 | 0 | 0 | **6** |
| Seeding | `CTRL041` – `CTRL049` | 1 | 3 | 5 | 0 | **9** |
| Execution | `CTRL050` – `CTRL052` | 2 | 1 | 0 | 0 | **3** |
| Developmental (Kelp FSPM) | `CTRL242` – `CTRL256` | 14 | 0 | 0 | 1 | **15** |
| **Total `panel_growth.cpp`** | | **47** | **6** | **12** | **2** | **67** |

### 2. `native/src/ui/panels/panel_styles.cpp` (146 catalog rows: 53 numbered + 93 style-editor)
Contains the growth presets library, record-new-preset toolchain, color themes, display typography, viewport element colors, and 3D lighting / shading controls. The catalog's Specimen Appearance section is no longer rendered here: CTRL104 moved to `panel_scene.cpp` (Scene > Layers), and CTRL105–CTRL106 are retired (`RETIRED-30` / `RETIRED-31`).

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Growth Presets Library | `CTRL053` – `CTRL056` | 0 | 0 | 3 | 1 | **4** |
| Record New Preset | `CTRL057` – `CTRL059` | 0 | 0 | 2 | 1 | **3** |
| Color Themes & Presets | `CTRL060` – `CTRL065` | 0 | 0 | 5 | 1 | **6** |
| Display Font / Typography | `CTRL066` – `CTRL070` | 2 | 1 | 1 | 1 | **5** |
| Viewport Element Colors | `CTRL071` – `CTRL086` | 0 | 0 | 1 | 15 | **16** |
| 3D Lighting & Shading ** | `CTRL087` – `CTRL103` | 16 | 0 | 1 | 0 | **17** |
| Specimen Appearance (CTRL105–CTRL106, retired) | `CTRL105` – `CTRL106` | 2 | 0 | 0 | 0 | **2** |
| **Total (numbered rows above)** | | **20** | **1** | **13** | **19** | **53** |

The 93 style-editor rows (`CTRL149`–`CTRL241`) also live in this file; they are described in the style-editor section below, giving the file's `Source_File` census of 146.

### 3. `native/src/ui/panels/panel_verification.cpp` (3 Controls Total)
Contains the verification log window controls.

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Selectables | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Verification Window | `CTRL109` – `CTRL111` | 0 | 0 | 2 | 1 | **3** |

### 4. `native/src/ui/panels/panel_scene.cpp` (40 controls)

| CSV Section | Control ID Range | Sliders | Toggles | Buttons | Combos / Pickers / Text | Section Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Verification Suite | `CTRL107` – `CTRL108` | 0 | 0 | 2 | 0 | **2** |
| Implicit Specimen | `CTRL112` – `CTRL115` | 1 | 0 | 3 | 0 | **4** |
| Layers Visibility | `CTRL116` – `CTRL126` | 0 | 11 | 0 | 0 | **11** |
| Specimen Appearance (CTRL104, rendered here) | `CTRL104` | 0 | 1 | 0 | 0 | **1** |
| Export Options | `CTRL127` – `CTRL138` | 5 | 2 | 4 | 1 | **12** |
| **Total `panel_scene.cpp`** | | **13** | **15** | **11** | **1** | **40** |

`Source_File` census across the four panels: 67 + 146 + 40 + 3 = 256. The build-time audit's `expected_source()` table special-cases `CTRL104` to `panel_scene.cpp`; the other rows follow the per-panel ranges above.

---

## 🔍 CONTROL INVENTORY BY FUNCTIONAL SECTION

### Section 1: Growth Command Bar (`panel_growth.cpp`)
- `CTRL001`: **Import** (`render_pill_button`) – Launches native file dialog to import 3D OBJ/STL specimen mesh.
- `CTRL002`: **Start** (`render_pill_button`) – Starts the growth bake with the Kelp Marine FSPM engine.
- `CTRL003`: **Stop** (`render_pill_button`) – Halts or pauses ongoing live root growth.
- `CTRL004`: **Clear** (`render_pill_button`) – Clears generated roots, resets growth simulation, and resets test objects.

### Section 2: Algorithm Selector & Parameters (`panel_growth.cpp`)
- `CTRL005`: **Algorithm Selector** (`begin_pill_combo`) – Selects among the algorithm profiles loaded in the session. This build ships one: Kelp Marine FSPM (engine mode 6). Import, export, switch and delete remain available.
- `CTRL006`: **Export Algorithm** (`ImGui::Button`) – Exports active algorithm definition to `.json` or `.cu`.
- `CTRL007`: **Import Algorithm** (`ImGui::Button`) – Imports custom algorithm definition file.
- `CTRL008`: **Delete Algorithm** (`ImGui::Button`) – Deletes active algorithm profile.
- `CTRL009`–`CTRL016` are retired catalog rows (`RETIRED-01` … `RETIRED-08`); they are registered and audited but have no `render_control()` call site.

### Section 3: Colonization Seeds (`panel_growth.cpp`)
- `CTRL017`: **Spore Count** (`render_compact_slider_int`, `2000` - `50000`, unit: count) – Number of nutrient attractor spores distributed inside vessel.
- `CTRL018`: **Primary Roots** (`render_compact_slider_int`, `1` - `8`, unit: count) – Number of primary root trunks sprouted from seed.
- `CTRL019`: **Search Radius** (`render_compact_slider_float`, `4.0` - `30.0`, unit: mm) – Maximum distance nodes can detect spores.
- `CTRL020`: **Kill Radius** (`render_compact_slider_float`, `0.5` - `10.0`, unit: mm) – Distance threshold at which spores are consumed.

### Section 4: Branching & Growth (`panel_growth.cpp`)
- `CTRL021`: **Variable Step** (`render_toggle_switch`) – Toggles adaptive step sizing along root length.
- `CTRL022`: **Min Growth Step** (`render_compact_slider_float`, `0.25` - `4.00`, unit: mm) – Lower bound for adaptive growth step.
- `CTRL023`: **Max Growth Step** (`render_compact_slider_float`, `0.25` - `6.00`, unit: mm) – Upper bound for adaptive growth step.
- `CTRL024`: **Growth Step Size** (`render_compact_slider_float`, `0.25` - `6.00`, unit: mm) – Fixed node segment step length.
- `CTRL025`: **Branching Rate** (`render_compact_slider_float`, `0.10` - `1.00`) – Frequency of lateral daughter branch splitting.
- `CTRL026` is a retired catalog row (`RETIRED-09`); it is registered and audited but has no live call site.
- `CTRL027`: **Max Steps** (`render_compact_slider_int`, `5` - `150`, unit: count) – Maximum growth iterations before auto-termination.

### Section 5: Vessel & Caliber (`panel_growth.cpp`)
- `CTRL028`: **Trunk Radius** (`render_compact_slider_float`, `0.10` - `3.00`, unit: mm) – Basal trunk radius at seed point.
- `CTRL029`: **Tip Radius** (`render_compact_slider_float`, `0.05` - `2.00`, unit: mm) – Terminal tendril radius at root tips.
- `CTRL030`: **Variable Taper** (`render_toggle_switch`) – Toggles continuous non-linear tapering.
- `CTRL031`: **Taper Curve** (`render_compact_slider_float`, `0.20` - `5.00`) – Exponent controlling taper profile curve.
- `CTRL032`: **Branch Smoothness (K)** (`render_compact_slider_float`, `0.00` - `2.00`, unit: mm) – SDF blend radius between branch junctions.
- `CTRL033`: **Wall Clearance** (`render_compact_slider_float`, `0.50` - `1.50`, unit: x) – Safe standoff distance multiplier from container boundary.
- `CTRL034`: **Wall Cut Smooth** (`render_compact_slider_float`, `0.00` - `2.00`, unit: mm) – Smoothness parameter for boolean boundary clipping.

### Section 6: Retired catalog rows (`panel_growth.cpp`)
- `CTRL035`–`CTRL040` are `RETIRED-10` … `RETIRED-15`; the six rows remain in the catalog and registry but are never rendered.

### Section 7: Seeding Management (`panel_growth.cpp`)
- `CTRL041`: **Seed Offset** (`render_compact_slider_float`, `0.00` - `5.00`, unit: mm) – Clearance offset from container interior surface.
- `CTRL042`: **Add Seed (+)** (`ImGui::Button`) – Dispatches GPU raycast to place a new seed inside specimen.
- `CTRL043`: **Remove Seed (-)** (`ImGui::Button`) – Deletes active seed and prunes all branches grown from it on GPU.
- `CTRL044`: **Freeze All Seeds** (`ImGui::Button`) – Locks current caliber on all seeds simultaneously.
- `CTRL045`: **Unfreeze All Seeds** (`ImGui::Button`) – Unlocks all seeds to respond to global caliber changes.
- `CTRL046`: **Select Seed (S1-Sn)** (`ImGui::Button`) – Selects active seed slot for interactive inspection and positioning.
- `CTRL047`: **Freeze Seed** (`render_toggle_switch`) – Locks individual seed radius.
- `CTRL048` and `CTRL049` are retired (`RETIRED-16`, `RETIRED-17`); they remain catalog/registry rows only and are never rendered.

### Section 8: Growth Execution (`panel_growth.cpp`)
- `CTRL050`: **Live Animated Growth** (`render_toggle_switch`) – Toggles continuous frame-by-frame growth simulation.
- `CTRL051`: **Step Rate (Live FPS)** (`render_compact_slider_int`, `1` - `30`, unit: fps) – Growth animation cadence limiter.
- `CTRL052`: **Steps Per Frame** (`render_compact_slider_int`, `1` - `10`, unit: count) – Number of growth algorithm iterations dispatched per frame.

### Section 9: Growth Presets Library (`panel_styles.cpp`)
- `CTRL053`: **Growth Preset Selector** (`begin_pill_combo`) – Selects active saved growth preset.
- `CTRL054`: **Apply Preset** (`ImGui::Button`) – Loads preset parameters into growth state.
- `CTRL055`: **Update Preset** (`ImGui::Button`) – Overwrites selected preset with active parameters.
- `CTRL056`: **Delete Preset** (`ImGui::Button`) – Removes selected preset from `imgui.ini`.
- `CTRL057`: **New Preset Name** (`ImGui::InputText`) – Input buffer for recording a custom preset.
- `CTRL058`: **Save New Preset** (`ImGui::Button`) – Commits new preset to persistent storage.
- `CTRL059`: **Export to Clipboard** (`ImGui::Button`) – Copies all preset configurations to Windows clipboard.

### Section 10: Color Themes & Layout Presets (`panel_styles.cpp`)
- `CTRL060`: **Theme Preset Selector** (`render_pill_combo`) – Switches theme (Modern Teal, Dark, Classic, Light).
- `CTRL061`: **Reset Theme** (`ImGui::Button`) – Restores Modern Dark Engineering teal palette.
- `CTRL062`: **Save Style Reference** (`ImGui::Button`) – Caches active ImGui style metrics.
- `CTRL063`: **Revert Style Reference** (`ImGui::Button`) – Restores cached style metrics.
- `CTRL064`: **Copy Style C++** (`ImGui::Button`) – Generates C++ code for active style colors to clipboard.
- `CTRL065`: **Copy Layout INI** (`ImGui::Button`) – Saves docking layout and styles to clipboard.

### Section 11: Display Font & Typography (`panel_styles.cpp`)
- `CTRL066`: **Display Font Selector** (`begin_pill_combo`) – Selects font family (Inter, Segoe UI, Roboto, etc.).
- `CTRL067`: **Bold Weight** (`render_toggle_switch`) – Toggles between Regular and Bold font weights.
- `CTRL068`: **Font Scale** (`render_compact_slider_float`, `0.50` - `2.00`, unit: x) – Global ImGui font scale multiplier.
- `CTRL069`: **Item Spacing Y** (`render_compact_slider_float`, `2.0` - `24.0`, unit: px) – Vertical UI element padding.
- `CTRL070`: **Reset Typography Settings** (`ImGui::Button`) – Restores default 1.0x scale and 4px padding.

### Section 12: Viewport Element Colors (`panel_styles.cpp`)
- `CTRL071` – `CTRL085`: **15 RGBA Color Pickers** (`ImGui::ColorEdit4`):
  - Viewport Background, Floor Grid, Floor X-Axis, Floor Y-Axis, Original Mesh, Body / Glass, Root Network, Branch Skeleton, Spores Cloud, Active Seed, Inactive Seeds, Bounding Box, Wireframe, Test Pass, Test Fail.
- `CTRL086`: **Reset Viewport Colors** (`ImGui::Button`) – Resets all 15 element colors to engineering defaults.

### Section 13: 3D Lighting & Shading (`panel_styles.cpp`)
- `CTRL087`: **Diffuse Light** (`render_compact_slider_float`, `0.20` - `1.50`) – Viewport directional light intensity.
- `CTRL088`: **Edge Darkening** (`render_compact_slider_float`, `0.00` - `1.00`) – Silhouette edge darkening intensity.
- `CTRL089`: **Crevice Occlusion** (`render_compact_slider_float`, `0.00` - `1.00`) – Curvature-based crevice darkening.
- `CTRL090`: **Specular** (`render_compact_slider_float`, `0.00` - `0.60`) – Blinn-Phong specular highlight strength.
- `CTRL091`: **RETIRED-28** (`render_compact_slider_float`, `0.00` - `2.00`) – Retired catalog row; the former relief-shading parameter is never rendered.
- `CTRL092`: **RETIRED-29** (`render_compact_slider_float`, `0.10` - `10.00`) – Retired catalog row; the former relief-shading frequency is never rendered.
- `CTRL093`: **Light Azimuth** (`render_compact_slider_float`, `-180.0` - `180.0`, unit: Degrees) – View-space key-light azimuth; the derived direction feeds both the GL mesh shader and the CUDA raymarcher.
- `CTRL094`: **Light Elevation** (`render_compact_slider_float`, `-89.0` - `89.0`, unit: Degrees) – View-space key-light elevation; the fill light follows at a fixed angular offset (`core/lighting_rig.hpp`).
- `CTRL095`: **Fill Light** (`render_compact_slider_float`, `0.00` - `1.00`) – Counter-fill weight in the base illumination expression (was hardcoded 0.16).
- `CTRL096`: **Ambient** (`render_compact_slider_float`, `0.00` - `1.00`) – Ambient floor in the base illumination expression (was hardcoded 0.32).
- `CTRL097`: **Glossiness** (`render_compact_slider_float`, `4.0` - `64.0`) – Blinn-Phong specular exponent (was hardcoded 24).
- `CTRL098`: **Highlight Tint** (`render_compact_slider_float`, `0.00` - `1.00`) – Specular highlight mix toward white (was hardcoded 0.55).
- `CTRL099`: **Fresnel Rim** (`render_compact_slider_float`, `0.00` - `1.00`) – Matte-glass fresnel rim gain (was hardcoded 0.30).
- `CTRL100`: **Volumetric AO (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – CUDA raymarch ambient-occlusion gain across the three stylized AO sites (was hardcoded 0.85).
- `CTRL101`: **SSS Strength (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – Subsurface transmittance strength through thin root walls (was hardcoded 0.45).
- `CTRL102`: **SSS Warmth (raymarch)** (`render_compact_slider_float`, `0.00` - `1.00`) – Warm tint mix of the subsurface scattering glow (was hardcoded 0.35).
- `CTRL103`: **Reset 3D Lighting** (`ImGui::Button`) – Resets all lighting coefficients to default values.

### Section 14: Transparent specimen (CTRL104, `panel_scene.cpp`)
The Styles panel's Specimen Appearance section was removed in v24.23. Its surviving toggle, `CTRL104` (Transparent specimen, bound to `display.specimen_transparent`), renders in Scene > Layers directly under Converted body (`CTRL117`). `CTRL105` and `CTRL106` are retired (`RETIRED-30`, `RETIRED-31`) and no longer rendered.

### Section 15: Verification Suite (`panel_scene.cpp` + `panel_verification.cpp`)
- `CTRL107`: **Run All Tests (Scene Panel)** (`ImGui::Button`) – Dispatches the full 19-test geometry verification harness on GPU.
- `CTRL108`: **Hide/View Log** (`ImGui::Button`) – Toggles visibility of floating verification results window.
- `CTRL109`: **Run Tests (Window)** (`ImGui::Button`) – Dispatches verification tests from within the tests window.
- `CTRL110`: **Clear 3D Tests** (`ImGui::Button`) – Removes test visual meshes from viewport and resets result list.
- `CTRL111`: **Load Test Model to Viewport** (`ImGui::Selectable`) – Loads a specific test result mesh as active specimen.

### Section 16: Implicit Specimen (`panel_scene.cpp`)
- `CTRL112`: **Detail Level Override** (`render_compact_slider_float`, `0.05` - `0.80`, unit: mm) – Voxel resolution for SDF rasterization.
- `CTRL113`: **Sphere (GPU Shape)** (`ImGui::Button`) – Synthesizes analytical 3D sphere mesh on GPU ($R = 20\text{ mm}$).
- `CTRL114`: **Torus (GPU Shape)** (`ImGui::Button`) – Synthesizes analytical 3D torus mesh on GPU ($R = 25\text{ mm}, r = 10\text{ mm}$).
- `CTRL115`: **Box (GPU Shape)** (`ImGui::Button`) – Synthesizes filleted box mesh on GPU ($40\text{ mm}$ with $4\text{ mm}$ fillet).

### Section 17: Layers Visibility (`panel_scene.cpp`)
- `CTRL116`: **Original Mesh** (`render_toggle_switch`) – Displays imported polygonal surface wireframe.
- `CTRL117`: **Converted Body** (`render_toggle_switch`) – Displays SDF reconstructed specimen body.
- `CTRL104`: **Transparent specimen** (`render_toggle_switch`) – Renders the converted body as semi-transparent glass; drawn here directly under Converted body (catalog section: Specimen Appearance).
- `CTRL118`: **Roots** (`render_toggle_switch`) – Displays solid 3D root network mesh.
- `CTRL119`: **Colorize Roots** (`render_toggle_switch`) – Renders branches with distinct per-seed colors.
- `CTRL120`: **Colorize Strands** (`render_toggle_switch`) – Renders individual tendrils with gradient color ramps.
- `CTRL121`: **Branch Skeleton** (`render_toggle_switch`) – Renders 3D centerline graph line segments.
- `CTRL122`: **Seeds** (`render_toggle_switch`) – Renders 3D spheres at root initiation seed positions.
- `CTRL123`: **Spores** (`render_toggle_switch`) – Renders 3D nutrient point cloud inside container.
- `CTRL124`: **Bounding Box** (`render_toggle_switch`) – Renders axis-aligned bounding box cage.
- `CTRL125`: **Generated Mesh (Wireframe)** (`render_toggle_switch`) – Renders polygon wireframe overlay on generated mesh.
- `CTRL126`: **Auto Orbit** (`render_toggle_switch`) – Activates turntable camera rotation.

### Section 18: Mesh Generation & Export (`panel_scene.cpp`)
- `CTRL127`: **Export Target** (`render_pill_combo`) – Merged Union, Roots Only, or Converted Body Only.
- `CTRL128`: **Adaptive Voxel Size** (`render_toggle_switch`) – Enables dual-resolution adaptive marching cubes.
- `CTRL129`: **Mesh Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Uniform marching cubes cell size.
- `CTRL130`: **Min Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Finest detail cell size in adaptive mode.
- `CTRL131`: **Max Voxel Size** (`render_compact_slider_float`, `0.05` - `0.60`, unit: mm) – Coarsest cell size in adaptive mode.
- `CTRL132`: **Smoothing Passes** (`render_compact_slider_int`, `0` - `15`, unit: count) – Number of GPU Laplacian smoothing iterations.
- `CTRL133`: **Smooth Strength** (`render_compact_slider_float`, `0.05` - `0.80`) – Weight multiplier per smoothing pass.
- `CTRL134`: **Manifold Repair** (`render_toggle_switch`) – Dispatches GPU topological repair to seal non-manifold edges.
- `CTRL135`: **Generate Mesh** (`ImGui::Button`) – Dispatches CUDA isosurface extraction and post-processing pipeline.
- `CTRL136`: **Export Mesh (OBJ)** (`ImGui::Button`) – Saves final mesh to Wavefront OBJ format.
- `CTRL137`: **Export Binary STL** (`ImGui::Button`) – Saves final mesh to binary 3D-printable STL format.
- `CTRL138`: **Export Binary PLY** (`ImGui::Button`) – Saves final mesh to binary Polygon File Format (PLY).

All ten rows are retired catalog markers. The procedural texture/displacement feature was removed in v24.23, so the ids, ranges and kinds stay registered and audited but have no `render_control()` call site and no `AppState` binding.
- `CTRL139`: **RETIRED-18** (`render_compact_slider_float`, `1.0` - `20.0`) – was Grain Scale.
- `CTRL140`: **RETIRED-19** (`render_compact_slider_float`, `1.0` - `10.0`) – was Vertical Stretch.
- `CTRL141`: **RETIRED-20** (`render_compact_slider_float`, `0.0` - `1.0`) – was Furrow Depth.
- `CTRL142`: **RETIRED-21** (`render_compact_slider_int`, `1` - `6`, unit: count) – was Noise Octaves.
- `CTRL143`: **RETIRED-22** (`ImGui::Button`) – was the 2D texture generator button.
- `CTRL144`: **RETIRED-23** (`render_compact_slider_float`, `-5.0` - `5.0`, unit: mm) – was Displacement (mm).
- `CTRL145`: **RETIRED-24** (`render_compact_slider_float`, `1.0` - `16.0`) – was Tile Repeat U.
- `CTRL146`: **RETIRED-25** (`render_compact_slider_float`, `1.0` - `32.0`) – was Tile Repeat V.
- `CTRL147`: **RETIRED-26** (`render_toggle_switch`) – was Recompute Normals.
- `CTRL148`: **RETIRED-27** (`ImGui::Button`) – was the mesh-displacement apply button.

---

## 🧩 STYLE EDITOR: REGISTERED

**Status:** the 93 style-editor widgets below are **inside** the 256-control catalog as `CTRL149`–`CTRL241` (three terminal CSV sections: Style Editor: Sizes / Colors / UI Rendering).

The Styles panel renders one block of widgets that is deliberately **outside** the numbered catalog: the **ADVANCED - INTERFACE STYLE EDITOR** parent pill (collapsed by default), containing three nested pills — **SIZES** (23 sliders: paddings, spacings, border sizes, rounding radii), **COLORS** (one `ImGui::ColorEdit4` per `ImGuiCol_` enumerator, the exact count determined by the ImGui version resolved through vcpkg), and **UI RENDERING (IMGUI)** (7 widgets: Global Alpha, Disabled Alpha, three anti-aliasing toggles, curve and circle tessellation tolerances).

These ~82 widgets are a Rhizome-skinned re-imagining of Dear ImGui's built-in `ShowStyleEditor()`, implemented as `render_style_editor_with_modern_preset()` in `native/src/ui/panels/panel_styles.cpp`. They bind **directly to `ImGuiStyle`** — the Dear ImGui toolkit's own interface state — rather than to `AppState`, through the `RHZ_BIND_IMGUI_STYLE_{F,F2,B,COL}` thunk family (the documented CTRL068/069 ImGui-global precedent, extended). Since v24.01 Phase 2 they are first-class catalog rows with three concrete enforcement consequences, all now in force: every widget has a `ControlDefinition` registry row, they are covered by the build-time and `--verify` CSV↔registry audits (plus the runtime census-coverage gate: the live Colors rows must equal the build's ImGuiCol_COUNT within the 63-row reference census (8d)), and they emit `ui_events.jsonl` telemetry — sliders/toggles through `render_control()`, the 63 census color pickers commit-only on `IsItemDeactivatedAfterEdit` (color drags fire per-frame; the event carries the drag-start and committed rgba values).

The three pills sit under the single ADVANCED parent pill (one stable section home) and the RENDERING pill keeps the **UI RENDERING (IMGUI)** name. Numbering is append-only — Sizes `CTRL149-171`, Colors `CTRL172-234`, UI Rendering `CTRL235-241` — so no domain id moved. The dynamic `ImGuiCol_COUNT` handling landed as a version-adaptive census (8c, re-pinned 8d): the 63 CTRL172-234 rows are INDEX-bound (no enumerator names), so any Dear ImGui with ImGuiCol_COUNT <= 63 builds cleanly — rows beyond the built count stay dormant (8d note: the census layout follows the 1.92.8+ enum order; where upstream inserted colors mid-enum, an older covered build's census labels past the insertion point follow the census layout, not that older enum's) — while newer builds are stopped by the `ImguiColCoverageCheck` compile gate (actual count printed; refresh via the one-time re-pin script (not kept in the tree)), with the `--verify` Test 19 `[Style Editor Registry]` gate reporting the census vs live counts and the `IMGUI_VERSION` string. Style changes now survive restarts: the Copy Layout INI exporter writes the full `[Style]` + `[StyleColors]` block and `load_imgui_style_from_ini` (panel_styles.cpp, called from main.cpp after `ImGui::CreateContext`) restores it.

---

## ⚙️ LOGGING SYSTEM INTEGRATION

Every UI interaction triggers a standardized call to `log_control_interaction()` (`native/src/ui/widgets/control_registry.hpp`), which resolves the exact CTRL-ID from the registry row and forwards to `RhizomeLog("ACTION", message)` (declared in `native/src/core/unified_logger.hpp`; implemented by the `UnifiedLogger` zero-alloc ring buffer with async flushing — console format unchanged):

```cpp
void RhizomeLog(const char* category, const char* message) {
    auto now = std::chrono::system_clock::now();
    // [HH:MM:SS.mmm] [CATEGORY] Message
    std::cout << "[" << time_str << "] [" << category << "] " 
              << (message ? message : "") << std::endl;
}
```

### Log Categories Mapped:
- **`ACTION`**: Slider adjustments, toggle flips, button clicks, and dropdown selections.
- **`REACTION`**: GPU execution metrics (e.g. `Mesh rebuilt in 23ms, triangles: 48,291`).
- **`MEM:BUFFER`**: Dynamic GPU memory allocations and buffer uploads.
- **`MEM:VRAM`**: Real-time snapshot of NVIDIA GPU VRAM occupancy.

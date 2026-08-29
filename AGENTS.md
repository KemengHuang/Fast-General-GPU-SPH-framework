# Agent Notes: Fast-General-GPU-SPH-framework

This document contains agent-focused context for working on the project.

## Project Overview

A GPU SPH (Smoothed Particle Hydrodynamics) framework that uses a uniform-grid neighbor search
with a hybrid SMS/TRA scheduling strategy. The codebase mixes C++ host code, CUDA device code,
GLUT/GLEW rendering, and JSON-based scene configuration.

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/main.cpp` | Common CLI entry point; dispatches compile-time headless or optional GUI mode |
| `src/` | All project source and headers |
| `src/core/` | Shared utilities: CUDA helpers, math, parameters, timers |
| `src/cuda_prescan/` | Prefix-sum helpers used by `grid/sph_arrangement.cu` (`scan.cu` is compiled; `prefix_sum.cu` is header-guarded and included by `scan.cu`) |
| `src/grid/` | Uniform-grid construction and particle sorting (`sph_arrangement`) |
| `src/io/` | GPU model loader/reader and statistics I/O |
| `src/particle/` | Particle buffer definitions and management (`particle_buffer`) |
| `src/render/` | Optional GLUT/GLEW GUI, camera state, screenshot, textures, CUDA-GL bridge |
| `src/simulation/` | High-level simulation class, marching cubes, PCISPH factor helpers |
| `src/solver/` | CUDA SPH kernels split by physics: density, force, integration, PCI-SPH, plus device context and host dispatch |
| `assets/` | Runtime JSON scenes and statistics files (`scene_default.json`, `*_statistics.json`, `insts_latency.json`, `ball32.png`) |
| `shaders/` | GL vertex/fragment shaders (currently `particle.vs` / `particle.fs`) |
| `third_party/lodepng/` | Third-party PNG loader |
| `data/` | Empty placeholder for runtime output |
| `build/` | CMake/Visual Studio build tree (ignored by git) |

## Build Environment

- **OS:** Windows 10/11
- **IDE:** Visual Studio 2022 Community
- **CUDA:** 12.x (tested with 12.4.131)
- **CMake:** >= 3.18
- **MSBuild:** Visual Studio 2022 toolset
- **vcpkg packages:** `jsoncpp`; GUI builds also need `glew` and `freeglut` (x64-windows)

### Configure

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

For a compute-only build with no GL dependencies or render sources:

```bash
cmake -S . -B build-headless -G "Visual Studio 17 2022" -A x64 ^
  -DGSPH_HEADLESS=ON
```

If vcpkg is not on the default search path, pass the toolchain file:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Build

```bash
cmake --build build --config Release
```

The executable is `build/Release/gsph.exe`. The VS startup project is set to `gsph` and the
debugger working directory is `$(OutDir)`.

## Runtime Assets

`CMakeLists.txt` always copies `assets/`. GUI builds also copy `shaders/`; headless builds do not.
All builds need the scene/statistics JSON files below; GUI builds additionally load
`ball32.png` and ship the shader directory:

- `scene_default.json`
- `scene_default1.json`
- `ball32.png`
- `insts_latency.json`
- `sph_sms_arti_block_statistics.json`
- `sph_tra_arti_block_statistics.json`
- `shaders/particle.vs`
- `shaders/particle.fs`

These paths are also hard-coded in:

- `src/simulation/sph_hybrid_system.cpp` (`kDefaultSceneFileName`, `ball32.png`)
- `src/io/gpu_model.cu` (statistics JSONs)

## Important Implementation Notes

- **`GSPH_HEADLESS=ON` is a compile-time graphics boundary.** CMake skips GL/GLEW/GLUT lookup
  and linking, excludes `src/render/` and lodepng, defines `GSPH_HEADLESS=1`, and omits all
  GL/CUDA-GL members and calls from `HybridSystem`. The headless executable defaults to a
  200-frame benchmark and accepts `--benchmark N` / `--headless N`.
- **CUDA textures were removed.** Older versions used `tex1Dfetch`; the current code reads
  particle data directly from global memory (`buff_list.position_d[idx]`,
  `buff_list.evaluated_velocity[idx]`). Do not reintroduce texture references.
- **Default scene is heavy.** `assets/scene_default.json` generates ~3.94M particles and may
  take a while to initialize. For quick iteration, use a smaller scene.
- **Grid-cell lookup uses precomputed reciprocals.** `ParticlePos2CellPos*` take
  `inv_cell_size` (multiplication), not `cell_size` (division); `SystemParameter::inv_cell_size`
  and `Arrangement::inv_cell_size_` are computed once at init.
- **`position_d.w` stores 1/density, not density.** The density pass writes the reciprocal
  (after the SMS-side clamp); the force pass and the integration kernels multiply by it
  (`a * w` instead of `a / w`), eliminating the per-neighbor-pair reciprocal. Any new consumer
  that needs the actual density must invert it at the use site.
- **Do not hand-pipeline the SMS register-path loops with prefetching**: it pushes the force
  kernel to 96 registers and roughly halves throughput (measured 2026-08-27). A 64-neighbor
  batch variant (`shared_pos[128]`) was also evaluated and reverted — no measurable win.
- **The production SMS layout is fixed at 32 particles per task and 64 threads per block.**
  The two warps stay independent; pairing them widens spatial bounds and is slower. CMake option
  `GSPH_USE_REGISTER_SMS` selects the register path (`ON`, default) or legacy shared A/B path.
- **Broken legacy kernels are kept but marked.** `knComputeDensitySMS/SMS64`,
  `knComputeForceSMS/SMS64` have uninitialized `cell_id` (assignments commented out) — launching
  them is undefined behavior. They are not on the live path; do not call them without restoring
  the missing pieces.
- **Screenshot capture runs before `glutSwapBuffers`** (back buffer is undefined after a swap)
  and creates the `screenshot/` directory on demand.
- **CMake recursively collects simulation sources under `src/`.** GUI builds add `src/render/`
  and `third_party/lodepng/`; headless builds filter both out. `prefix_sum.cu` is header-guarded
  and is pulled in by `scan.cu`; the glob only makes it visible in the IDE tree.
- **Unused dependencies were removed from CMake.** `Eigen3`, `cublas`, `cusparse`, and
  `cusolver` are not linked anymore because they are not used in the source.

## Coding Conventions

- Line endings are normalized to LF (see `.gitattributes`).
- Indent with 4 spaces (see `.editorconfig`).
- Source files should be UTF-8.
- Keep comments in English; remove or translate garbled/non-ASCII comments.
- Prefer minimal changes; large kernel-file splits and renderer refactors are out of scope for
  light cleanups.

## Performance Notes

- **CUDA architecture is pinned to the local GPU.** `CMakeLists.txt` sets
  `CMAKE_CUDA_ARCHITECTURES` to `89` for the RTX 4090 workstation. Reconfigure after pulling
  changes so the cache entry is updated. Pass `-DCMAKE_CUDA_ARCHITECTURES=120` explicitly for
  the RTX 5090 benchmark workstation.
- **MSVC `/O2` is only added in Release/RelWithDebInfo.** Debug builds keep the default `/Od`
  and `/RTC1`; this avoids the "/O2 and /RTC1 are incompatible" error when building the
  `Debug` configuration in Visual Studio.
- **CUDA-GL interop is used for rendering.** `HybridSystem::drawParticles()` registers the
  position/color VBOs with CUDA and copies `final_position`/`color` directly from device memory
  into the VBOs with a small kernel (`copyParticleDataToVBOs`). This avoids the previous
  synchronous D2H + `glBufferData` host round-trip.
- **Per-frame CUDA event creation was removed.** Timing events are created once and reused.
  The detailed timing path still synchronizes on the last event, so turn off
  `get_detailed_time_` for maximum throughput.
- **The leftover benchmark file** (`combine666...txt`) and its locked file handle were removed;
  the file is no longer opened at startup.
- **Per-frame scalar D2H copies are now asynchronous.** `middle_value_` and `h_num_cta_` are
  copied into pinned host buffers with `cudaMemcpyAsync` + `cudaStreamSynchronize`, avoiding the
  implicit global device sync of synchronous `cudaMemcpy`.
- **Hybrid kernel launch sizing.** The density/force hybrid kernels read the TRA/SMS split
  point (`d_middle_value_`) and the SMS task count (`d_num_cta_`) directly from device memory
  and size their own branch split; `HYBRID_DEVICE_GRID_SIZING` in `src/core/sph_utils.cuh`
  selects the host-side strategy. `0` (default) keeps one pinned async readback + stream sync
  per frame and launches exact grids; `1` over-provisions the grids and skips the sync.
  Measured on RTX 4090 / 3.94M particles, `0` is ~2% faster (over-provisioned block scheduling
  costs more than the sync).
- **`__launch_bounds__(64, 10)` is enabled on the hybrid kernels** and validated spill-free
  on RTX 5090 / `sm_120` (register path density: 39 regs, force: 72 regs; no stack/local).
  Recheck with `cuobjdump -res-usage` after kernel edits or when changing the target architecture.
- **Particle buffers are sized by exact particle count.** `initializeScene` counts the fluid
  blocks before allocating; `recomm_nump` in the scene JSON is only a fallback. (Previously
  `recomm_nump: 15500000` over-allocated ~1.9 GB of device/pinned memory for the default
  3.94M-particle scene.)
- **Velocity/acceleration limiters are proper clamps.** `knIntegrateVelocity*` clamp
  `|v|`/`|a|` to `sqrt(limit)` (`limit` is the squared-magnitude bound). Before, they scaled
  by a constant `1/sqrt(limit)`, crushing any over-limit velocity to a fixed small value.
- **Benchmark mode no longer syncs per frame.** `runBenchmark` collects per-stage CUDA-event
  timings from 5 warm-up frames and measures wall-clock FPS over the remaining frames with a
  single final `cudaDeviceSynchronize`.
- **Detailed timing is off by default.** Set `get_detailed_time_ = true` in
  `src/simulation/sph_hybrid_system.h` only when you need per-stage timing; it still inserts a
  per-frame `cudaEventSynchronize`.

## Common Issues

- If Visual Studio has the `.vs` database open, `rm -rf build` may fail. Reconfigure in place
  with `cmake -S . -B build ...` instead.
- Missing DLLs (`freeglut.dll`, `glew32.dll`, `jsoncpp.dll`) must be placed next to the
  executable or on `PATH`.

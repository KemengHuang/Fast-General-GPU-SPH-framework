# Agent Notes: Fast-General-GPU-SPH-framework

This document contains agent-focused context for working on the project.

## Project Overview

A GPU SPH (Smoothed Particle Hydrodynamics) framework that uses a uniform-grid neighbor search
with a hybrid SMS/TRA scheduling strategy. The codebase mixes C++ host code, CUDA device code,
GLUT/GLEW rendering, and JSON-based scene configuration.

## Repository Layout

| Path | Purpose |
|------|---------|
| `src/` | All project source and headers |
| `src/cuda_prescan/` | Prefix-sum helpers used by `sph_arrangement.cu` (`scan.cu` is compiled; `prefix_sum.cu` is header-guarded and included by `scan.cu`) |
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
- **vcpkg packages:** `glew`, `freeglut`, `jsoncpp` (x64-windows)

### Configure

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
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

`CMakeLists.txt` copies `assets/` and `shaders/` to the output directory as a post-build step.
The executable expects the following files in its working directory:

- `scene_default.json`
- `scene_default1.json`
- `ball32.png`
- `insts_latency.json`
- `sph_sms_arti_block_statistics.json`
- `sph_tra_arti_block_statistics.json`
- `shaders/particle.vs`
- `shaders/particle.fs`

These paths are also hard-coded in:

- `src/sph_hybrid_system.cpp` (`kDefaultSceneFileName`, `ball32.png`)
- `src/gpu_model.cu` (statistics JSONs)

## Important Implementation Notes

- **CUDA textures were removed.** Older versions used `tex1Dfetch`; the current code reads
  particle data directly from global memory (`buff_list.position_d[idx]`,
  `buff_list.evaluated_velocity[idx]`). Do not reintroduce texture references.
- **Default scene is heavy.** `assets/scene_default.json` generates ~3.94M particles and may
  take a while to initialize. For quick iteration, use a smaller scene.
- **`CMakeLists.txt` recursively collects all source/header files under `src/` and
  `third_party/lodepng/`.** `prefix_sum.cu` is header-guarded and is pulled in by
  `scan.cu`; adding it via the recursive glob only makes it visible in the IDE tree.
- **Unused dependencies were removed from CMake.** `Eigen3`, `cublas`, `cusparse`, and
  `cusolver` are not linked anymore because they are not used in the source.

## Coding Conventions

- Line endings are normalized to LF (see `.gitattributes`).
- Indent with 4 spaces (see `.editorconfig`).
- Source files should be UTF-8.
- Keep comments in English; remove or translate garbled/non-ASCII comments.
- Prefer minimal changes; large kernel-file splits and renderer refactors are out of scope for
  light cleanups.

## Common Issues

- If Visual Studio has the `.vs` database open, `rm -rf build` may fail. Reconfigure in place
  with `cmake -S . -B build ...` instead.
- Missing DLLs (`freeglut.dll`, `glew32.dll`, `jsoncpp.dll`) must be placed next to the
  executable or on `PATH`.

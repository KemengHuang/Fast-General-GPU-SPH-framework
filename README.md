# Fast-General-GPU-SPH-framework

This framework is a CUDA/C++17 implementation of Smoothed Particle Hydrodynamics (SPH) using a
uniform-grid neighbor search and hybrid SMS/TRA scheduling.

## Description

This project is the source code of
["Novel Hierarchical Strategies for SPH-centric Algorithms on GPGPU"](https://doi.org/10.1016/j.gmod.2020.101088)
and
["A General Novel Parallel Framework for SPH-centric Algorithms"](https://dl.acm.org/doi/10.1145/3321360).

It provides a research implementation and benchmark for uniform-grid GPU SPH scheduling,
neighbor exchange, and kernel optimization.

Source code contributors: [Kemeng Huang](https://kemenghuang.github.io), Jiming Ruan.

## Build & Run

### Requirements

- CMake >= 3.18
- CUDA Toolkit 12.x or 13.x (tested with CUDA 12.4 and 13.2)
- A C++17 toolchain (tested with Visual Studio 2022/2026 on Windows)
- jsoncpp; GUI builds additionally require GLEW and FreeGLUT. For example, via
  [vcpkg](https://vcpkg.io) (pick the triplet matching your platform):
  ```bash
  vcpkg install glew freeglut jsoncpp --triplet x64-windows
  ```

For a headless-only machine, `jsoncpp` is the only vcpkg dependency; OpenGL, GLEW, GLUT,
render sources, shaders, and lodepng are not included in that build.

**Platform note:** out-of-the-box builds target Windows. The timer, screenshot helper, GL setup,
and parts of the host code require platform adaptation before building on Linux or macOS.

### Configure

Standard single-config generators (Ninja, Makefiles):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Visual Studio (multi-config):

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

Compute-only headless build:

```bash
cmake -S . -B build-headless -G "Visual Studio 17 2022" -A x64 \
  -DGSPH_HEADLESS=ON
```

If the dependencies are not on the default search path, point CMake at vcpkg:

```bash
cmake -S . -B build ... -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
```

### CMake options

| Option | Default | Purpose |
|---|---:|---|
| `GSPH_HEADLESS` | `OFF` | Exclude OpenGL, GLUT/GLEW, lodepng, shaders, and render sources |
| `GSPH_USE_REGISTER_SMS` | `ON` | Select the optimized register SMS path; `OFF` builds the shared-memory reference |
| `GSPH_ENABLE_HISTORICAL_ISSAME` | `ON` | Preserve historical `judgeTask` pairing and GUI task colors |
| `GSPH_ENABLE_SMS_LOCAL_MERGE` | `ON` | Cooperatively stage `isSame` pairs during density |
| `GSPH_ENABLE_SMS_LOCAL_MERGE_FORCE` | `OFF` | Experimental paired force staging; slower on RTX 5090 |
| `GSPH_ENABLE_SAME_CELL_PAIR_FORCE` | `OFF` | Research prototype for pair-once same-cell force; substantially slower |
| `GSPH_ENABLE_IPO` | `ON` | Enable host IPO and CUDA device LTO in optimized configurations |

The default CUDA architecture is `89`. Pass `-DCMAKE_CUDA_ARCHITECTURES=120` for a native RTX
5090 build. Example optimized headless configuration:

```bash
cmake -S . -B build-headless -G "Visual Studio 17 2022" -A x64 \
  -DGSPH_HEADLESS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
```

### Build

```bash
cmake --build build --config Release   # --config is only needed for multi-config generators
```

Or open the generated solution/project in your IDE and build the `Release` target. The
executable is `gsph` (`gsph.exe` on Windows) under `build/` or `build/Release/`, and the
runtime `assets/` directory is copied next to it automatically. GUI builds also copy `shaders/`;
headless builds do not compile or copy graphics assets.

### Run

Launch the simulation from the build output directory (so the bundled assets are found):

```bash
build/Release/gsph.exe   # Windows / Visual Studio
# or
./build/gsph             # single-config generators
```

The default scene (`assets/scene_default.json`) generates ~3.94 million particles. For faster
iteration during development, create a smaller scene file and change `kDefaultSceneFileName` in
`src/simulation/sph_hybrid_system.cpp`.

### Headless benchmark

`GSPH_HEADLESS=ON` produces an executable with no OpenGL imports or render code. It runs 200
frames by default; override the count with `--benchmark N` or `--headless N`:

```bash
build-headless/Release/gsph --benchmark 200
```

The normal GUI build also accepts the same benchmark arguments and bypasses window creation at
runtime, but it remains linked to the graphics libraries. Use the CMake option when graphics
dependencies must be absent entirely. Headless kernels also omit the render-only color and
`final_position` updates; GUI builds retain both.

This prints wall-clock FPS, the TRA/SMS split, a state checksum, and per-stage
timings (grid arrange / density / force). `--headless N` is an alias.

### Optimized SMS path

- One 32-particle task is assigned to each warp; production blocks contain 64 threads.
- `BlockTask` caches coarse-cell coordinates, cell begin/count, particle offset, and historical
  paired bounds, avoiding repeated integer division and cell-array loads in the physics kernels.
- `position_d.w` stores reciprocal density, eliminating a reciprocal from each force-neighbor
  interaction.
- Density cooperatively stages eligible historical `isSame` pairs. Force retains independent
  warp traversal because paired force staging reduced throughput.
- CUB scans and radix sorts share one persistent device temporary allocation sized at startup.
  Persistent key/value alternates remove runtime sort allocations; no Thrust headers or calls
  remain in the source tree.
- The exact-grid launch path remains the default. The device-sized over-provisioned alternative
  is available through `HYBRID_DEVICE_GRID_SIZING` in `src/core/sph_utils.cuh`, but measured slower.

The optimized register SMS path is enabled by default. Configure an equivalent legacy shared
path build for A/B testing with:

```bash
cmake -S . -B build-shared -DGSPH_USE_REGISTER_SMS=OFF
```

The register build uses the project's historical `judgeTask` policy. Each globally aligned SMS
task pair `(0,1), (2,3), ...` gets `isSame=1` exactly when both tasks have the same coarse
`cellid`; `judgeTask` stores the historical combined bounds in the first descriptor's packed
padding and handles an odd tail task. The original per-task bounds remain hot for independent
register traversal, and there is no additional search-volume heuristic.

Density uses cooperative staging for `isSame` pairs by default. Force keeps the original task
bounds and stays on the faster independent register iterator. The controls are:

```bash
-DGSPH_ENABLE_HISTORICAL_ISSAME=OFF       # disable classification and task coloring
-DGSPH_ENABLE_SMS_LOCAL_MERGE=OFF         # keep isSame/colors, disable density staging
-DGSPH_ENABLE_SMS_LOCAL_MERGE_FORCE=ON    # experimental force staging (slower on RTX 5090)
-DGSPH_ENABLE_SAME_CELL_PAIR_FORCE=ON     # pair-once research prototype (substantially slower)
```

Benchmark output reports the number and shape of paired tasks. In GUI builds the historical
density-pass coloring is preserved: `isSame` SMS tasks are cyan, independent SMS tasks are
yellow, and TRA particles remain magenta.

### Controls

- `Space` – pause / resume
- `w`/`s` – move forward / backward
- `a`/`d` – move left / right
- `q`/`e` – move down / up
- `o`/`u` – increase / decrease particle point size
- Arrow keys – move the light source
- `t` – advance one simulation step while paused (debug)
- `/` – toggle screenshot capture to `screenshot/`

## Project Layout

```
.
├── assets/          runtime JSON scenes and textures
├── shaders/         GL vertex/fragment shaders
├── src/             source code
│   ├── main.cpp         common CLI/headless entry point
│   ├── core/            shared utilities (CUDA helpers, math, parameters, timers)
│   ├── cuda_prescan/    legacy recursive prescan helpers; the live path uses persistent CUB
│   ├── grid/            uniform-grid construction and particle sorting
│   ├── io/              GPU model loader/reader and statistics I/O
│   ├── particle/        particle buffer definitions and management
│   ├── render/          optional GLUT/GLEW GUI, excluded from headless builds
│   ├── simulation/      high-level simulation, marching cubes, PCISPH helpers
│   └── solver/          CUDA SPH kernels split by physics, plus dispatch
├── third_party/     GUI-only third-party code (lodepng)
├── agent_docs/      code analysis, optimization reports, known issues (dev notes)
├── data/            placeholder for runtime output
├── CMakeLists.txt
└── README.md
```

## Performance

Reference headless Release results for `assets/scene_default.json` (~4M particles):

| GPU / CUDA | Variant | Mean frame | FPS |
|---|---|---:|---:|
| RTX 4090 / CUDA 12.4 / `sm_89` | Historical optimized register build | ~28.6 ms | ~35.0 |
| RTX 5090 / CUDA 13.2 / `sm_120` | Shared-memory reference | 22.901 ms | 43.67 |
| RTX 5090 / CUDA 13.2 / `sm_120` | Register path (default) | **21.833 ms** | **45.80** |

The RTX 5090 values are the means of two interleaved 1000-frame runs per variant. The register
path delivered 4.89% higher throughput (4.66% lower frame time). Replacing the remaining Thrust
code with persistent CUB storage measured 22.792 ms versus 22.835 ms before the change in a
separate interleaved 750-frame A/B, effectively no online regression.

Performance varies with driver state, clocks, scene evolution, and background GPU load. Use
interleaved runs and compare means rather than selecting a single best result. See
`agent_docs/optimization_report.md` for the complete history and rejected experiments.

## Validation

The current source has been verified with:

- Register headless Release (`GSPH_USE_REGISTER_SMS=ON`)
- Shared-memory headless Release (`GSPH_USE_REGISTER_SMS=OFF`)
- GUI Release with CUDA-OpenGL interop
- Compute Sanitizer `memcheck` and `synccheck` on both register and shared builds: 0 errors
- `cuobjdump -res-usage` on native `sm_120`: density 39/38 registers, force 67 registers, no
  stack/local spills

Example sanitizer invocation:

```bash
compute-sanitizer --tool memcheck build-headless/Release/gsph.exe --benchmark 6
compute-sanitizer --tool synccheck build-headless/Release/gsph.exe --benchmark 6
```

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE).

## BibTex

Please cite the following papers if this work helps your research.

```
@article{HUANG2020101088,
  title = {Novel hierarchical strategies for SPH-centric algorithms on GPGPU},
  journal = {Graphical Models},
  volume = {111},
  pages = {101088},
  year = {2020},
  issn = {1524-0703},
  doi = {https://doi.org/10.1016/j.gmod.2020.101088},
  url = {https://www.sciencedirect.com/science/article/pii/S152407032030028X},
  author = {Kemeng Huang and Zipeng Zhao and Chen Li and Changbo Wang and Hong Qin}
}
```

```
@article{10.1145/3321360,
  author = {Huang, Kemeng and Ruan, Jiming and Zhao, Zipeng and Li, Chen and Wang, Changbo and Qin, Hong},
  title = {A General Novel Parallel Framework for SPH-Centric Algorithms},
  year = {2019},
  issue_date = {May 2019},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {2},
  number = {1},
  url = {https://doi.org/10.1145/3321360},
  doi = {10.1145/3321360},
  journal = {Proc. ACM Comput. Graph. Interact. Tech.},
  month = {jun},
  articleno = {7},
  numpages = {16}
}
```

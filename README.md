# Fast-General-GPU-SPH-framework

This framework is a fast, general implementation of a GPU SPH method utilizing a uniform grid.

## Description

This project is the source code of
["Novel Hierarchical Strategies for SPH-centric Algorithms on GPGPU"](https://doi.org/10.1016/j.gmod.2020.101088)
and
["A General Novel Parallel Framework for SPH-centric Algorithms"](https://dl.acm.org/doi/10.1145/3321360).

It offers fast optimization strategies based on a uniform grid. It also serves as an excellent benchmark for further research on GPU SPH and for meaningful comparisons.

Source code contributors: [Kemeng Huang](https://kemenghuang.github.io), Jiming Ruan.

**Note: this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation.**

## Build & Run

### Requirements

- CMake >= 3.18
- CUDA Toolkit 12.x
- A C++17 toolchain (MSVC 2022, GCC, or Clang)
- jsoncpp; GUI builds additionally require GLEW and FreeGLUT. For example, via
  [vcpkg](https://vcpkg.io) (pick the triplet matching your platform):
  ```bash
  vcpkg install glew freeglut jsoncpp --triplet x64-windows   # or x64-linux, ...
  ```

For a headless-only machine, `jsoncpp` is the only vcpkg dependency; OpenGL, GLEW, GLUT,
render sources, shaders, and lodepng are not included in that build.

**Platform note:** the code currently has a few Windows-only pieces (`windows.h`-based timers,
`CreateDirectoryA` in the screenshot helper, backslash-style GL includes), so out-of-the-box
builds target Windows. Porting to Linux/macOS only requires small shims in those spots.

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

The repository default is `sm_89`. Override it for another GPU, for example
`-DCMAKE_CUDA_ARCHITECTURES=120` on RTX 5090.

Release and RelWithDebInfo builds enable host IPO and CUDA device LTO by default. Disable it for
toolchain compatibility or A/B testing with `-DGSPH_ENABLE_IPO=OFF`.

If the dependencies are not on the default search path, point CMake at vcpkg:

```bash
cmake -S . -B build ... -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake
```

### Build

```bash
cmake --build build --config Release   # --config is only needed for multi-config generators
```

Or open the generated solution/project in your IDE and build the `Release` target. The
executable is `gsph` (`gsph.exe` on Windows) under `build/` or `build/Release/`, and the
runtime assets (`assets/`, `shaders/`) are copied next to it automatically.

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
```

Benchmark output reports the number and shape of paired tasks. In GUI builds the historical
density-pass coloring is preserved: `isSame` SMS tasks are cyan, independent SMS tasks are
yellow, and TRA particles remain magenta.

The project contains no Thrust dependency. CUB scans and radix sorts share persistent device
temporary storage allocated once by `Arrangement`; key/value alternate buffers are persistent as
well. The optional `GSPH_ENABLE_SAME_CELL_PAIR_FORCE=ON` research prototype is intentionally off
because shared-atomic pair accumulation is substantially slower on the default dense scene.

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
│   ├── cuda_prescan/    prefix-sum helpers included by grid/sph_arrangement.cu
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

Reference numbers for the default scene (~3.94M particles), measured with
`gsph --benchmark 200` on an RTX 4090 / CUDA 12.4:

- **~35 FPS** (28.6 ms/frame): ~1.2 ms grid arrange, ~9.5 ms density, ~20 ms force

On RTX 5090 / CUDA 13.2 / native `sm_120`, the final two interleaved 1000-frame runs per variant
measured 22.901 ms/frame for the shared path and 21.833 ms/frame for the register path
(~4.89% throughput improvement, or ~4.66% lower frame time).

See `agent_docs/optimization_report.md` for the measured optimization history, including
evaluated-and-rejected experiments (device-side grid sizing, neighbor-batch prefetching).

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

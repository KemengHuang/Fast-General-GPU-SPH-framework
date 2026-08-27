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
- GLEW, FreeGLUT and jsoncpp — e.g. via [vcpkg](https://vcpkg.io) (pick the triplet matching
  your platform):
  ```bash
  vcpkg install glew freeglut jsoncpp --triplet x64-windows   # or x64-linux, ...
  ```

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

Run a fixed number of frames without opening a window (useful for profiling and CI-style
performance checks):

```bash
gsph --benchmark 200
```

This prints wall-clock FPS plus per-stage timings (grid arrange / density / force) collected
from CUDA events. `--headless N` is an alias.

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
│   ├── core/            shared utilities (CUDA helpers, math, parameters, timers)
│   ├── cuda_prescan/    prefix-sum helpers included by grid/sph_arrangement.cu
│   ├── grid/            uniform-grid construction and particle sorting
│   ├── io/              GPU model loader/reader and statistics I/O
│   ├── particle/        particle buffer definitions and management
│   ├── render/          GLUT/GLEW renderer, camera, screenshot, textures
│   ├── simulation/      high-level simulation, marching cubes, PCISPH helpers
│   └── solver/          CUDA SPH kernels split by physics, plus dispatch
├── third_party/     third-party code (lodepng)
├── agent_docs/      code analysis, optimization reports, known issues (dev notes)
├── data/            placeholder for runtime output
├── CMakeLists.txt
└── README.md
```

## Performance

Reference numbers for the default scene (~3.94M particles), measured with
`gsph --benchmark 200` on an RTX 4090 / CUDA 12.4:

- **~35 FPS** (28.6 ms/frame): ~1.2 ms grid arrange, ~9.5 ms density, ~20 ms force

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

# Fast-SPH-framework

This framework is a fast, general implementation of a GPU SPH method utilizing a uniform grid.

## Description

This project is the source code of
["Novel Hierarchical Strategies for SPH-centric Algorithms on GPGPU"](https://doi.org/10.1016/j.gmod.2020.101088)
and
["A General Novel Parallel Framework for SPH-centric Algorithms"](https://dl.acm.org/doi/10.1145/3321360).

It offers fast optimization strategies based on a uniform grid. Compared to a well-optimized GPU SPH method using the uniform grid, the proposed approach achieves a speed improvement of up to 3.5x. It therefore serves as an excellent benchmark for further research on GPU SPH and for meaningful comparisons.

Source code contributors: [Kemeng Huang](https://kemenghuang.github.io), Jiming Ruan.

**Note: this software is released under the MPLv2.0 license. For commercial use, please email the authors for negotiation.**

## Build & Run

### Requirements

- Windows 10/11
- Visual Studio 2022 Community (or higher)
- CUDA Toolkit 12.x
- CMake >= 3.18
- vcpkg with the following packages installed:
  ```bash
  vcpkg install glew freeglut jsoncpp --triplet x64-windows
  ```

### Configure

Open a terminal in the repository root and run:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

Make sure CMake can find vcpkg, e.g. by setting `CMAKE_TOOLCHAIN_FILE`:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### Build

Open `build/Hybrid_Fluid_Simulation.sln` in Visual Studio and build the `Release` target,
or build from the command line:

```bash
cmake --build build --config Release
```

The executable will be produced at `build/Release/gsph.exe`.

### Run

CMake copies the required runtime assets (`assets/`) and shaders (`shaders/`) to the output
directory automatically. Launch the simulation with:

```bash
build/Release/gsph.exe
```

The default scene (`assets/scene_default.json`) generates ~3.94 million particles. For faster
iteration during development, create a smaller scene file and change `kDefaultSceneFileName` in
`src/sph_hybrid_system.cpp`, or add a command-line argument (not implemented yet).

### Controls

- `Space` – pause / resume
- `w`/`s` – move forward / backward
- `a`/`d` – move left / right
- `q`/`e` – move down / up
- `o`/`u` – increase / decrease particle point size
- `/` – toggle screenshot capture to `screenshot/`

## Project Layout

```
.
├── assets/          runtime JSON scenes and textures
├── shaders/         GL vertex/fragment shaders
├── src/             source code
│   └── cuda_prescan/    prefix-sum helpers included by sph_arrangement.cu
├── third_party/     third-party code (lodepng)
├── CMakeLists.txt
└── README.md
```

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

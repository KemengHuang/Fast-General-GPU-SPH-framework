# Code Analysis — Fast-General-GPU-SPH-framework

Snapshot: commit `efb8c0a` (branch `test`). Line references below refer to the **pre-optimization**
state (commit `e6afc64`) unless noted otherwise.

## 1. Architecture Overview

GPU SPH fluid solver using a uniform-grid neighbor search with a hybrid **SMS/TRA** scheduling
strategy. C++ host code + CUDA device code + fixed-function GLUT/GLEW rendering + JSON scene
files. Target: Windows / VS2022 / CUDA 12.x / RTX 4090 (sm_89).

### Per-frame live path (`HybridSystem::tick`, `src/simulation/sph_hybrid_system.cpp`)

| Stage | What runs |
|---|---|
| Arrange | `Arrangement::arrangeHybridFrame()` (`src/grid/sph_arrangement.cu`): micro-cell histogram → CUB scan → physical particle reorder → per-cell info → task requirement model → compaction index → task-offset scan → `knArrangeIndependentSmsTasks` → one scalar D2H/sync → `padIndependentSmsTasks` |
| Density | `computeDensityHybridKernel` — one fused launch; TRA branch gathers sparse particles through `compact_indices`, while the SMS branch runs two independent 32-particle warps per 64-thread block using `SmsRegisterTaskIterator` |
| Force | `computeForceHybridKernel` — same split and task structure; neighbor velocity is read only after the distance test passes |
| Integrate | `knIntegrateVelocityE` |
| Render | `knCopyToVBOs` writes `final_position`/`color` directly into CUDA-registered GL VBOs (no host round-trip) |

Neighbor search: cell size = kernel radius `h`, so 27 neighbor cells; both paths iterate 9 "rows"
(y,z ∈ {-1,0,1}) and merge 3 x-adjacent cells into one contiguous index range (valid because the
sort key is x-major). The "M" refinement uses a 4× finer micro-grid (64 micro-cells per cell) so
partial x-ranges at row ends are counted from micro-cell offsets.

Everything outside this path — plain SMS/SMS64/TRA kernels, PCISPH, multiphase, marching cubes —
is **dead at runtime** (~40% of the source tree).

## 2. Module Notes

### `src/solver/` — CUDA kernels
- Hybrid kernels are the only live compute kernels. Physics constants broadcast via
  `__constant__ SystemParameter kDevSysPara`.
- `knComputeDensitySMS/SMS64`, `knComputeForceSMS/SMS64` are **broken** (`cell_id` uninitialized,
  staging calls commented out) — now marked with `WARNING` comments; do not launch them.
- `pcisph_kernels.cu` is 271 lines of empty stubs; the `predictionCorrectionStep*` family
  hardcodes `max_predicted_density = 1000.0f` so its convergence test degenerates.
- `kernel_common.cuh` still contains many legacy shared-memory helper classes; the live register
  path uses the single `SmsRegisterTaskIterator` implementation for both density and force.

### `src/grid/sph_arrangement.cu` — sorting / task generation
- ~1600 lines with ≥8 dead arrange variants and large commented debug dumps.
- The live SMS path keeps adjacent warp tasks independent. `padIndependentSmsTasks` only creates
  one inactive descriptor for an odd tail, and the task buffer uses an exact safe upper bound.
- `gpu_model.cu` heuristic `(nump_self + 27) >> 5` under-allocates SMS tasks for cells with
  33–36 particles (tail particles keep stale density for that frame) — an upstream heuristic
  quirk, not a regression from this work.

### `src/particle/` — buffers
- SoA, three `ParticleBufferObject`s: `device_buff_` (72 B/particle), `device_buff_temp_`
  (48 B, aliasing base for two fields), `host_buff_` (pinned, 44 B).
- `position_d.w` doubles as density, `evaluated_velocity.w` as pressure (undocumented packing).
- `velocity`/`acceleration`/`final_position` are 12-byte `float3` — misaligned 3-word accesses.

### `src/render/`
- 100% fixed-function pipeline; `shaders/particle.vs/.fs` are never compiled or loaded.
- Point sprites via `GL_POINT_SPRITE_ARB` + `ball32.png` texture; HUD via `glutBitmapCharacter`.

## 3. Issues Found and Fixed (this pass)

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| A1 | `cuda_call_check.h:22,34` | `exit(0)` before `std::cerr` — every CUDA error was a silent, success-looking exit | print first, then `exit(EXIT_FAILURE)` |
| A2 | `particle_buffer.h:127` | deleted `operator=(const ParticleBufferList&)` — wrong type, real copy-assign still compiled → device-pointer double-free | delete `operator=(const ParticleBufferObject&)` |
| A3 | `sph_hybrid_system.h:37` | `Scene::x/y/z` uninitialized when JSON lacks `xx/yy/zz` (e.g. `scene_default1.json`) | default-initialize to 0 |
| A4 | `sph_hybrid_system.cpp:142` | `3 / 4` integer division → `self_lplc_color` always 0 (field is unused) | kept for behavior, documented in comment |
| A5 | `sph_hybrid_system.cpp` dtor | `arrangement_` never deleted — leaked all grid buffers | `delete arrangement_` |
| A6 | `sph_hybrid_system.cpp:245` | lazy tick-event creation leaked stream + 2 events on re-entry | `createPersistentCudaResources` now idempotent |
| A7 | `sph_arrangement.cu:363` | thread-0 `*mid_val = -1` raced the transition write; a lost race silently made a frame all-TRA | `cudaMemsetAsync(-1)` before launch; in-kernel init removed |
| A8 | `device_context.cu:19` | host `printf` read the host stub of a `__constant__` symbol (garbage) every parameter upload | removed |
| A9 | `integration_kernels.cu:292-302,120-129` | limiters compared **squared** magnitude to the limit, then scaled by constant `1/sqrt(limit)` — any over-limit |v| was crushed to a fixed small value | proper clamp: `v *= sqrt(limit / speed²)` (behavior change, approved) |
| A10 | `main.cpp` + `screenshot.cpp` | screenshot read the back buffer **after** `glutSwapBuffers` (undefined) and `screenshot/` was never created | capture before swap; `CreateDirectoryA` on demand |
| A11 | `main.cpp:130` vs `:279` | near plane 10.0 in `init()` vs 0.001 in `reshape()` — resize silently changed clipping | unified to 0.1 |
| A12 | `gl_texture.h` | uninitialized members; default copy double-freed `data_` | member init, `default:` case, copy ops deleted |
| A13 | `main.cpp:124` | `glewInit()` return unchecked | check `GLEW_OK`, abort with message |

## 4. Remaining Landmines (not fixed — dead code, kept per scope decision)

- The four broken legacy kernels (see 2.1) are still callable through their dispatch wrappers.
- `ParticleBufferList` carries 13 pointer fields that are never allocated (PCI-SPH/multiphase
  leftovers); any revival of those paths null-dereferences.
- `sph_arrangement.cu:616,647,649` — `hashp`/`d_p_offset_`/`d_p_offset_p` are allocated with
  `nump_` while `d_hash_`/`d_index_` use capacity; `resetNumParticle` grows only the latter.
  Enabling particle insertion past the initial count is a device heap overflow.
- `pcisph_factor.cpp` computes the PCI factor from uninitialized device memory; `exit(0)` in
  library code; per-call malloc/cudaMalloc + synchronous full-array D2H.
- `sph_marching_cube.cpp` — per-call `new float[w*h*d]`, O(nump×9³) CPU loop, ASCII STL output;
  grid-stride mismatch (`iso_radius/4` vs `kernel/4`) mis-scales meshes; keep out of frame loop.
- `scan.cu` legacy recursive prescan is dead on the live path (CUB replaced it); float/int
  variants share `g_numEltsAllocated`, so using both trips an assert.

# Optimization Report

Date: 2026-08-27 · Machine: RTX 4090, CUDA 12.4, VS2022 · Scene: `assets/scene_default.json`
(3,944,312 particles) · Method: `gsph.exe --benchmark 200` (headless)

## Results

| Stage | Avg frame | FPS | Notes |
|---|---|---|---|
| Baseline (`e6afc64`) | 32.68 ms | 30.6 | arrange 1.25 / density 10.15 / force 20.69 / kernel total 32.63 |
| After correctness fixes | 29.02 ms | 34.5 | limiter fix changed physics → particle distribution changed; not strictly comparable |
| After kernel optimizations | 28.77 ms | 34.8 | |
| After buffer right-sizing | 28.56 ms | 35.0 | ~1.9 GB less device+pinned allocation at startup |
| Final config | **28.6 ± 0.2 ms** | **~35.0** | run-to-run noise ≈ ±2% |

Caveat: the A9 limiter fix (proper velocity/acceleration clamps) changed the simulated workload,
so the baseline↔final delta (~12.5%) mixes behavior change with optimization. The pure
kernel-level optimizations contribute roughly 2–4%.

## What was changed

### Correctness (13 items)
See `code_analysis.md` §3 for the full table. The highest-impact ones: CUDA error macros that
silently exited with code 0; the wrong deleted `operator=` signature; the limiter clamp bug;
the `Arrangement` shutdown leak; the `knFindHybridModeMiddleValue` write race.

### Kernel optimizations (behavior-preserving)
- **`__launch_bounds__(64, 10)` restored** on both hybrid kernels — validated spill-free via
  `cuobjdump -res-usage`: density 40 regs, force 57 regs, 0 stack/local.
- **Dead `isSame == 1` branches removed** from the live kernels (`isSame` is hard-wired to 0);
  the original shared-memory path remains reachable via `DENSITY_SMS_USE_REGISTER_PATH 0` /
  `FORCE_SMS_USE_REGISTER_PATH 0`.
- **`inv_cell_size` precomputed** (`SystemParameter::inv_cell_size`, `Arrangement::inv_cell_size_`);
  `ParticlePos2CellPos*` now multiply instead of dividing per thread. All call sites updated.
- **Redundant `cell_offset[cell_id]` / `cell_num[cell_id]` loads hoisted** into registers in the
  SMS branches.
- **`size_t` → `int` loop counters** on the live inner loops (no 64-bit address arithmetic).
- Integration kernels reuse the already-loaded `evaluated_velocity` register value.

### Pipeline / host
- Hybrid kernels can read the TRA/SMS split (`d_middle_value_`) and SMS task count
  (`d_num_cta_`) from device memory and size their own branch split, guarded by
  `HYBRID_DEVICE_GRID_SIZING` in `src/core/sph_utils.cuh`. **Default is 0** (see negative
  result below).
- `runBenchmark` no longer syncs per frame: per-stage CUDA-event timings come from 5 warm-up
  frames; the measured loop runs without per-frame `cudaEventSynchronize` and drains once at the
  end, so wall-clock FPS reflects true throughput.
- Per-frame `HighResolutionTimerForWin` construction hoisted to a member; HUD FPS uses the
  smoothed wall-clock value (division-by-zero on `total_time_` removed).

### Memory
- `initializeScene` counts the fluid blocks exactly before allocating; `recomm_nump` is only a
  fallback. Default scene: 15.5M → 3.94M capacity, saving ~1.9 GB device+pinned and noticeable
  startup time. `assets/scene_default.json`/`scene_default1.json` hints updated accordingly.
- `scan.cu` `deallocBlockSumsInt` checked the float pointer — fixed (legacy path, dead at runtime).

## Negative result (kept, documented, off by default)

Eliminating the mid-frame `cudaStreamSynchronize(0)` (which read back two scalars to size the
`judgeTask`/density/force launches) was implemented with device-side grid sizing: kernels read
`d_middle`/`d_num_block` themselves, grids are conservatively over-provisioned, excess blocks
exit immediately.

Measured A/B on the default scene:

| Variant | Avg frame |
|---|---|
| Host sync + exact grids (`HYBRID_DEVICE_GRID_SIZING 0`) | 28.0–28.3 ms |
| Device sizing + over-provisioned grids (`1`) | 28.6–28.8 ms |

The ~200k over-provisioned blocks cost more to schedule than the single sync saved — the GPU
stays busy draining the launch queue during the sync, so the sync is nearly free here. The
device-side capability remains available via the macro; on other scenes or weaker CPUs it may
still win.

## Reproduce

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
build/Release/gsph.exe --benchmark 200
```

Register check after kernel changes:

```bash
cuobjdump -res-usage build/gsph.dir/Release/density_kernels.obj
```

## Not done (deliberate scope cuts)

- No dead-code mass deletion (empty kernels, PCI-SPH stubs, legacy arrange variants).
- No renderer modernization (fixed-function pipeline is load-bearing).
- No `inv_density` array to replace the per-neighbor-pair reciprocal — traffic/ALU trade-off
  unclear; revisit only if force remains the bottleneck after profiling.
- No physical-scatter removal (SMS shared-memory staging depends on cell-contiguous layout).

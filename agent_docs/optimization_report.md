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

## SMS neighbor-exchange A/B: register path vs legacy shared-memory path

Measured 2026-08-27 (GPU idle, 2×200-frame runs per variant, default scene, averages):

| Variant | FPS | density | force | regs (den/for) | shared (den/for) |
|---|---|---|---|---|---|
| Register path (`*_SMS_USE_REGISTER_PATH 1`, default) | 34.4 | ~10.1 ms | ~19.8 ms | 40 / 57 | 1632 / 2656 B |
| Legacy shared path (`0`) | 34.7 | ~9.5 ms | ~20.0 ms | 39 / 54 | 1632 / 2656 B |

**No measurable difference** (all deltas within ±3% run-to-run noise). Both variants stage
neighbors through a 64×`float4` shared buffer in 32-particle batches with identical global-memory
traffic (`__ldg` per neighbor); the register path only changes (a) where the loaded value lands
first (register vs shared) and (b) barrier granularity (`__syncwarp` vs 2× `__syncthreads` per
batch). Since shared usage, occupancy, and memory traffic are identical, and barrier stalls are
hidden by the many other resident blocks per SM, neither wins on this scene. The register path
could still matter with high warp-level work imbalance (the two warps of a block run different
tasks; `__syncthreads` couples them) or lower SM occupancy. Keep the register path as default —
equivalent here, structurally safer against imbalance.

## SMS register-path experiments (2026-08-27, second pass)

Four candidate optimizations of the register-exchange SMS path, each measured with 2×200-frame
benchmark runs on the idle GPU (default scene, 3.94M particles). Baseline for this round:
34.4 FPS / density 10.06 ms / force 19.84 ms.

| # | Change | Result | Verdict |
|---|--------|--------|---------|
| V3 | Full-unroll fast path for full batches (`read_num == batch`) in `knComputeCell*Reg64` | density 10.06 → 9.14 ms, force flat | reverted (user decision: keep the code simple) |
| V2 | Neighbor batch 32 → 64 per warp (`shared_pos[128]`, two loads per lane, half as many state-machine iterations) | density → 8.82 ms, force flat, FPS flat | reverted (user decision) |
| V1 | Software-pipelined prefetch of the next batch during compute | **21.3 FPS — big regression**; force kernel jumped 57 → 96 regs (occupancy 53% → 31%) | reverted |
| V4 | `position_d.w` stores `1/density`; force/integrate multiply instead of dividing (removes one MUFU.RCP per neighbor pair) | 35.1–35.4 FPS, density → 8.65 ms, force ~flat | **kept** |

Final state = V4 only. Confirmed with the reverts applied: 35.34 / 34.95 FPS (2×200 frames),
density ~9.8–10.2 ms, force ~19.7 ms, force kernel back to 57 regs / 1632 B shared (density 40
regs). Total FPS matches the V3+V2+V4 combination within noise — the frame is force-dominated,
so the V2/V3 density-stage gains (~1.2 ms) did not translate into visible FPS. Keeping V4 alone
is therefore both simpler and equally fast. Lessons: the register path's cost is the per-pair
compute, not the exchange or the batch bookkeeping; hand-written prefetching is
counterproductive at this register budget; and wider batches (64) trade more shared traffic per
batch for fewer bookkeeping iterations with no net win.

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

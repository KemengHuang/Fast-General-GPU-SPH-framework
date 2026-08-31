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
- **The old `isSame == 1` branch was removed** from the live kernels at this stage. The historical
  `judgeTask` behavior was later restored at the user's request; see "Historical judgeTask SMS
  pairing" below. Configure with `-DGSPH_USE_REGISTER_SMS=OFF` for the legacy shared-memory A/B
  build.
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

## Register/shared final A/B and cleanup (2026-08-29)

RTX 5090, CUDA 13.2, native `sm_120`, default 3,944,312-particle scene, three interleaved
200-frame runs:

| Variant | Mean frame | Relative throughput |
|---|---:|---:|
| Legacy shared iterator (`GSPH_USE_REGISTER_SMS=OFF`) | 23.009 ms | 1.000× |
| Register iterator (`ON`, default) | 21.112 ms | 1.090× |

Both builds were spill-free. The register build used 39 registers for density and 72 for force;
Compute Sanitizer reported zero errors. A diagnostic build measured about 5,172 candidate pairs
but only 1,174 interacting neighbors per particle/frame, explaining why storage-only changes
cannot plausibly produce a 1.5× whole-frame speedup without changing the neighbor algorithm.

The production path is now fixed at 32 particles per task and 64 threads per block. Density and
force share `SmsNeighborIterator`; the duplicated legacy register iterator classes and the
obsolete `apply_reg.py`/`.register_attempt` artifacts were removed.

## Historical judgeTask SMS pairing (2026-08-30)

The `isSame` decision matches `origin/old` and commit `efb8c0a`. `judgeTask` examines globally
aligned task pairs `(0,1), (2,3), ...`; matching `cellid` values are the only eligibility
condition. It sets both descriptors to `isSame=1`, applies the historical bounds rule (second x
maximum; second z maximum when `first.xxi == second.xxx`, otherwise z=`[0,3]`), and pads an odd
tail. There is deliberately no search-volume threshold or alternative compatibility heuristic.

The implementation now assigns one `judgeTask` thread per pair instead of launching no-op odd
threads. The 32-byte `BlockTask` retains the original bounds, stores the six packed 2-bit pair
bounds, and caches cell coordinates/begin/count. Density consumes the packed bounds for
cooperative `isSame` staging; force defaults to the original bounds and an independent register
iterator. Force staging remains available for A/B but was slower on RTX 5090.

Stabilized 2×500-frame runs measured approximately 22.77 ms with density staging, 22.87 ms with
classification/coloring but no staging, and 22.91 ms with `isSame` disabled. The small density
gain is retained while the larger force-stage regression is removed.

Enabling CMake IPO/CUDA device LTO reduced two 500-frame runs from about 22.55 ms to 22.10 ms
(~2.0%) on RTX 5090 / CUDA 13.2. `GSPH_ENABLE_IPO=ON` is therefore the optimized-build default;
it can be disabled for compatibility or A/B testing.

The old scheduler visualization is restored as well: merged SMS particles are cyan, independent
SMS particles are yellow, and TRA particles remain magenta.

The build at that checkpoint remained spill-free on RTX 5090 / `sm_120`; later resource and
performance numbers are superseded by the final pass below.

## Final measured optimization pass (2026-08-30)

The following behavior-preserving changes survived isolated A/B tests:

- Precomputed scene-invariant density/force coefficients removed the per-particle reciprocal of
  `rest_density` and repeated constant products. Two interleaved 1000-frame pairs were about 1.0%
  faster overall.
- The task-requirement kernel now uses linear ±x/±y/±z offsets and fixed 32-particle shifts instead
  of repeated 3D index conversion/runtime division. Its static SASS fell from 200 to 128
  instructions; the measured whole-frame gain was ~0.36%. Empty scheduling cells return early.
- `CountingSort_Result_M` no longer writes a temporary coarse-cell offset that the next kernel
  immediately reads and overwrites. Nsight Systems measured its median at 286.5 → 270.9 μs; the
  combined sort/classification saving was about 15 μs/frame.
- `middle_value` and SMS task count now share one device/pinned `int2`, reducing two 4-byte
  `cudaMemcpyAsync` submissions to one 8-byte submission (~5.7 μs/frame of host API time).
- Headless builds omit render-only `final_position` and color updates. The integration-kernel
  median improved from 273.0 to 233.4 μs; GUI builds still generate positions and preserve
  cyan/yellow/magenta scheduler colors.

Rejected in this pass: multiplication-tree `powf_7` (neutral/slightly slower), fused task
classification (whole-frame ~0.19% slower despite a smaller arrangement event), and a persistent
device-sized grid (22.84–28.57 ms versus 20.74 ms for the exact-grid control). All three were
removed from production source.

Final RTX 5090 / CUDA 13.2 / native `sm_120` result, two interleaved 1000-frame runs per variant:

| Variant | Mean frame | Relative throughput |
|---|---:|---:|
| Legacy shared iterator (`GSPH_USE_REGISTER_SMS=OFF`) | 22.901 ms | 1.000× |
| Register iterator (`ON`, default) | **21.833 ms** | **1.0489×** |

This is a measured 4.89% throughput improvement (4.66% lower frame time), not the previously
requested 1.5×. The final headless register kernels are spill-free: mixed/SMS-only density uses
39/38 registers and 2336 B shared; force uses 67 registers and 3328 B shared. Final Compute
Sanitizer memcheck and synccheck runs reported zero errors for both register and shared builds.

## Requested algorithm/CUB pass (2026-08-31)

Four higher-risk candidates were implemented, built and measured on the same RTX 5090 default
scene before deciding whether they belonged in production:

| Candidate | Measured result | Verdict |
|---|---:|---|
| X-plane-aligned variable SMS tasks for tighter micro-cell bounds | tasks 130,953 → 160,692; ~24–25 ms | Reverted: partial-lane/task overhead exceeded pruning savings |
| Same-cell particle pair computed once with shared accumulation for both particles | force ~27.7 ms; frame ~35.5 ms | Off by default: shared atomic reduction dominated |
| One 32-thread force task per CUDA block | 22.833 vs 22.862 ms (~0.13%) and 72 regs | Reverted: noise-level gain and lower available warp count |
| Integration writes the next frame's hash/counts | 23.242 vs 22.743 ms (~2.19% slower) | Reverted: contended atomics lengthened the integration critical path |

The Thrust removal was retained. `thrust::sort`, the host round-trip `sort_by_key`, all Thrust
headers, and stale Thrust comments were removed. CUB `DeviceScan`, `SortKeys`, and `SortPairs`
share one persistent device temporary allocation sized once at initialization; persistent key and
value alternates avoid runtime allocations. Interleaved 750-frame runs measured 22.792 ms after
the change versus 22.835 ms before it (~0.19%, effectively no online regression). Final register
and shared memcheck/synccheck runs again reported zero errors.

## Reproduce

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
build/Release/gsph.exe --benchmark 200
```

For the shared-memory A/B build, add `-DGSPH_USE_REGISTER_SMS=OFF` at configure time.
For isSame coloring with fully independent register traversal, add
`-DGSPH_ENABLE_SMS_LOCAL_MERGE=OFF`. To disable classification/coloring too, add
`-DGSPH_ENABLE_HISTORICAL_ISSAME=OFF`.
For a build with no OpenGL/render dependencies, add `-DGSPH_HEADLESS=ON`.

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

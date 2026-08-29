# Known Issues & Future Work

Tracked from the 2026-08-27 analysis/optimization pass (commit `efb8c0a`).
Items are ordered by severity, not by priority.

## Correctness landmines (dead code, live triggers)

1. **Broken legacy SMS kernels are still launchable.** `knComputeDensitySMS/SMS64`
   (`src/solver/density_kernels.cu`) and `knComputeForceSMS/SMS64` (`src/solver/force_kernels.cu`)
   have `cell_id` uninitialized and staging calls commented out — launching them is undefined
   behavior. Marked with `WARNING` comments. Fix properly or delete with their dispatch wrappers.
2. **Unallocated `ParticleBufferList` fields** (`src/particle/particle_buffer.h`):
   `predicted_pos`, `correction_pressure_force`, `pressure`, `predicted_density`, `densityError`,
   `correction_pressure`, `phase`, `vlfrt`, `mixV`, `Vm`, `mixP`, `mixM`, `Mixden` have no
   allocation anywhere, but dead PCI-SPH/multiphase dispatch paths dereference them. Any revival
   of those paths null-dereferences. Either allocate in the PCI-SPH path or strip the fields.
3. **Particle insertion overflows device buffers** if enabled: `hashp`, `d_p_offset_`,
   `d_p_offset_p` are allocated with `nump_` (not capacity) and `Arrangement::resetNumParticle`
   grows only `d_hash_`/`d_index_` (`src/grid/sph_arrangement.cu:616,647,649,1478`). The
   `action1` path also reads `sys_para_.spacing_fluid`, which is commented out of parameter init.
4. **PCI-SPH factor** (`src/simulation/pcisph_factor.cpp`) reads uninitialized device memory,
   does a per-call malloc + cudaMalloc + synchronous full-array D2H + serial host reduction, and
   calls `exit(0)` on a zero divisor. Dead code; implement or remove.
5. **Marching cubes** (`src/simulation/sph_marching_cube.cpp`) is O(nump×9³) on CPU per call,
   leaks `density` on early return, has a grid-stride mismatch (`iso_radius/4` for sampling vs
   `kernel/4` for polygonization), and a `vindex > w*h*d` bound that should be `>=`.
6. **`judgeTask` odd-count tail** writes a fabricated task at `block_tasks[numb]`
   (`src/grid/sph_arrangement.cu:937`) — valid only because `d_block_task_` capacity is
   `numc * 10`. Document or assert the ≤320-particles/cell assumption.
7. **`gpu_model.cu` hard-coded block counts** (`:244-247`: literal 5,5,7,7) disagree with the
   statistics JSON if it changes; `recommended_times` is uninitialized for other counts and
   uploaded to the GPU. The mangled kernel names at `:25-28` must byte-match the JSONs.
8. **`scan.cu` global state**: float/int scan variants share `g_numEltsAllocated` (double use
   asserts); `deallocBlockSumsInt` pointer check was fixed, the rest of the legacy prescan is
   unused on the live path.

## Performance ideas (evaluated, deferred)

- **`inv_density` array**: replace the per-neighbor-pair `1 / neighbor_pos.w` (one MUFU.RCP per
  pair) with a per-particle reciprocal stored by the density pass. Costs +4 B/pair of L1 traffic;
  unclear win on a memory-bound kernel. Profile first.
- **Micro-cell table cost**: the 38 MB memset + 10.08M-int CUB scan per frame could be restricted
  to the occupied cell window (needs a small reduction for min/max occupied cell). Partial win
  (~44% of the table for the default scene), medium complexity.
- **Physical particle reorder** (~380 MB/frame) could in principle be replaced by the
  already-built index indirection, but the SMS shared-memory staging depends on cell-contiguous
  layout — architecture-level change.
- **Device-side grid sizing** (`HYBRID_DEVICE_GRID_SIZING 1`) currently loses ~2% to null-block
  scheduling. Could be revisited with grid-stride kernels (fixed grid, exact device count) instead
  of over-provisioned grids.
- **`float3` → `float4`** for `velocity`/`acceleration`/`final_position` would fix misaligned
  3-word accesses at +4 B/particle. Touchy change (many load/store sites), modest expected win.
- **GL state hoisting**: ~20 constant GL state calls per frame in `drawParticles`; per-frame
  window-title `sprintf`+`glutSetWindowTitle`. Cosmetic CPU cost, no GPU impact.

## Cleanup debt (out of scope last pass)

- ~40% of the tree is dead code: empty kernels/stubs in `pcisph_kernels.cu`, large parts of
  `force_kernels.cu`/`integration_kernels.cu`, ~8 dead arrange variants, ~1700 lines of unused
  shared-data classes in `kernel_common.cuh`, the whole `cuda_prescan` module (CUB replaced it),
  `initializeScene2`/`addParticle2`/`action1`, marching cubes, unused shaders.
- `knCalculateBlockRequirementHybridMode` (`src/io/gpu_model.cu:165`) uses `(nump_self + 27) >> 5`
  in one branch, which under-allocates one SMS task for cells with 33–36 particles (tail particles
  keep stale density for the frame). Upstream heuristic quirk; changing it alters published
  behavior.
- The optional GUI build still uses fixed-function GL; `shaders/particle.vs/.fs` are shipped but
  never loaded. Headless builds exclude the complete graphics subtree and shader copy step.

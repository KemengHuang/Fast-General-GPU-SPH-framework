// Auto-generated declarations for density_kernels.cu.

#ifndef _SOLVER_DENSITY_KERNELS_CUH_
#define _SOLVER_DENSITY_KERNELS_CUH_

#include "solver/kernel_common.cuh"

namespace sph {

__device__
inline void knComputeCellDensitySMS64(const int& isSame, SimDenSharedData128 *sdata, CDAPData *self_data, int read_num);
__device__
inline void knComputeCellDensitySMS(SimDenSharedData *sdata, CDAPData *self_data, int read_num);
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeDensitySMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task);
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeDensitySMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task);
__device__
void knComputeCellDensityTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos);
__device__
void knComputeCellDensityTRA9(ParticleBufferList &buff_list, CDAPData *self_data, int cell_offset, int cell_num);
__device__
float knComputeCellMixDensityTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos);
__global__
void knComputeDensityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range);
__global__
void knComputeMixDensityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range);
void launchDensityHybridKernel(
    int block_count, bool sms_only, int *micro_cell_offsets,
    ParticleIdxRange tra_range, ParticleBufferList buffers,
    int *compact_indices, int *cell_offsets, int *cell_particle_counts,
    const BlockTask *block_tasks, const int *device_sms_task_count,
    const int *device_middle);

}

#endif

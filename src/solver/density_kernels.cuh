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
__global__ //__launch_bounds__(64, 10)
void kncomputeDensityHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset);

}

#endif

//
// gpu_model.cuh
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _GPU_MODEL_CUH
#define _GPU_MODEL_CUH

namespace gpu_model
{

struct GPUModel;

void allocateGPUModel(GPUModel *&gm);

void freeGPUModel(GPUModel *gm);

void calculateBlockRequirementSMSMode(int *block_req, int *cell_start, int *cell_end, int block_size, int numc);

void calculateBlockRequirementHybridMode(int *cell_type, int *d_cell_num, int *block_req, GPUModel *gm, int *cell_offset, int *cell_num, ushort3 grid_size, int block_size);

}

#endif/*_GPU_MODEL_CUH*/

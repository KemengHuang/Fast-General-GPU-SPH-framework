//
// sph_kernel.cuh
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _SPH_KERNEL_CUH
#define _SPH_KERNEL_CUH

#include <cuda_runtime.h>
#include "sph_parameter.h"
#include "sph_particle.h"
#include "pcisph_factor.h"

namespace sph
{

struct ParticleIdxRange // [begin, end), zero-based numbering
{
    __host__ __device__
    ParticleIdxRange(){}
    __host__ __device__
    ParticleIdxRange(int b, int e) : begin(b), end(e) {}
    int begin, end;
};


void BuffInit(ParticleBufferList buff_list_n, int nm);

void transSysParaToDevice(const SystemParameter *host_para);

void initializeKernel();

void releaseKernel();

void find_max_P(int blocks, int tds, sumGrad *id_value, int numbers);



void computeMixDensityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void computeDriftVelocityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void computeVolumeFracTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void computeAccelTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void advanceMix(ParticleBufferList buff_list, int nump);
















void computeDensityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void computeForceTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void computeDensitySMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeDensitySMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);
void computeDensityHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeForceHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);
//void computeDensityHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

//void computeForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeOtherForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeOtherForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block);

void computeOtherForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeOtherForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);


void computeOtherForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeOtherForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeOtherForceTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num);

void manualSetting(ParticleBufferList buff_list, int nump, int step);

void advance(ParticleBufferList buff_list, int nump);
void advanceWave(ParticleBufferList buff_list, int nump, float time);
//sf pcisph-----------------------

void advancePCI(ParticleBufferList buff_list, int nump);

float computeDensityErrorFactorTRA(float mass, float rest_density, float time_step, ParticleBufferList buff_list, int *cell_offset, int *cell_num, uint nump);


void computeGradWValuesSimpleSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, sumGrad *particle_device);

void computeGradWValuesSimpleTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range, sumGrad *particle_device);

void predictionCorrectionStepSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, 
							float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed);





void predictionCorrectionStepTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block
                                  , float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float pcisph_max_density_error_allowed);





void predictionCorrectionStepSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                 float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed);


void predictionCorrectionStepHybrid(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                    float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range);

void predictionCorrectionStepHybrid128(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                    float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range);

void predictionCorrectionStepHybrid128n(ParticleBufferList buff_list_n,int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                       float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range);

void predictionCorrectionStepTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, 
                                 float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range);



void predictPositionAndVelocity(ParticleBufferList buff_list, unsigned int nump);
void computePredictedDensityAndPressureSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);



void computePredictedDensityAndPressureTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);



void computePredictedDensityAndPressureSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);

void computePredictedDensityAndPressureHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);

void computePredictedDensityAndPressureHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);

void computePredictedDensityAndPressureHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block, float pcisph_density_factor);
void computePredictedDensityAndPressureTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range, float pcisph_density_factor);




void getMaxPredictedDensityCUDA(ParticleBufferList buff_list, float& max_predicted_density, unsigned int nump);
void computeCorrectivePressureForce(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);


void computeCorrectivePressureForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);



void computeCorrectivePressureForce64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeCorrectivePressureForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeCorrectivePressureForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);
void computeCorrectivePressureForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeCorrectivePressureForceTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range);

//sf heat conduction-------------------
void computeHeatFlux(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

void computeTemperatureAndPhaseTransAndGetVis(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block);

}

#endif/*_SPH_KERNEL_CUH*/

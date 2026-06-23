// Auto-generated from src/sph_kernel.cu during solver reorganization.
// Host-side wrappers and initialization helpers.

#include "solver/kernel_dispatch.cuh"
#include "solver/device_context.cuh"
#include "solver/density_kernels.cuh"
#include "solver/force_kernels.cuh"
#include "solver/integration_kernels.cuh"
#include "solver/pcisph_kernels.cuh"
#include "simulation/pcisph_factor.h"
#include "core/cuda_call_check.h"

#include <cuda_runtime.h>

namespace sph {

void BuffInit(ParticleBufferList buff_list_n, int nm){
    if (nm <= 0) return;
    int num_thread = 256;
    int number_block = ceil_int(nm, num_thread);
    knInit << <number_block, num_thread >> >(buff_list_n, nm);
}

void computeDensitySMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 2);
    int num_thread = 64;
    knComputeDensitySMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_num, block_task);


}

void computeDensitySMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_thread = 32;

    knComputeDensitySMS << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, block_task);

}
void computeDensityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    knComputeDensityTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}







void computeMixDensityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);
    knComputeMixDensityTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
    //CUDA_SAFE_CALL(cudaEventRecord(tra_density_event));
}
void computeForceTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    knComputeForceTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);

}

void computeDriftVelocityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num){
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);
    knComputeDriftVelocityTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}
void computeVolumeFracTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num){
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);
    kncomputeVolumeFracTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}

void computeAccelTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num){
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);
    kncomputeAccelTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}

void computeForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 2);
    int num_thread = 64;
  
    knComputeForceSMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_num, block_task);
  
}

void computeForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_thread = 32;

    //knComputeOtherForceSMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_number, block_task);
    knComputeForceSMS << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, block_task);

}

void computeOtherForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;


    int num_thread = kDefaultNumThreadSMS;

    knComputeOtherForceSMS << <num_block, num_thread >> >(buff_list, cell_offset, cell_number, block_task);


}

void computeOtherForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 3);

    int num_thread = 96;//kDefaultNumThreadSMS;
    knComputeOtherForceTRAS << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_number, block_task);
}



void computeOtherForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 2);

    int num_thread = 64;// kDefaultNumThreadSMS;

    knComputeOtherForceSMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_number, block_task);

}
void computeOtherForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block){

    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    /* std::cout << num_block << std::endl;;
    std::cout << bt_offset << std::endl;;
    std::cout << number_blocks;*/

    knComputeOtherForceHybrid128 << <number_blocks, num_thread >> >(range, buff_list, cell_offset, cell_number, block_task, bt_offset);

}
void computeDensityHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){

    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;
//	std::cout << ceil_int(num_block, 2) << "               " << num_block << "         asfasdfasfasdfafsd" << std::endl;
	kncomputeDensityHybrid128n << <number_blocks, num_thread >> >(cell_offset_M, range, buff_list_n, cindex, cell_offset, cell_num, block_task, bt_offset);
}

void computeForceHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){
    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;
	kncomputeForceHybrid128n << <number_blocks, num_thread >> >(cell_offset_M,range, buff_list_n, cindex, cell_offset, cell_num, block_task, bt_offset);
}

void computeOtherForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int* cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){

    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    /* std::cout << num_block << std::endl;;
    std::cout << bt_offset << std::endl;;
    std::cout << number_blocks;*/

    knComputeOtherForceHybrid128n << <number_blocks, num_thread >> >(range, buff_list_n, cindex, cell_offset, cell_num, block_task, bt_offset);

}


void computeOtherForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block){

    int total_thread = range.end - range.begin;
    int num_thread = 32;
    int bt_offset = 0;
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, num_thread);
        num_block += bt_offset;
    }
    if (num_block <= 0) return;

    knComputeOtherForceHybrid << <num_block, num_thread >> >(range, buff_list, cell_offset, cell_number, block_task, bt_offset);

}

void computeOtherForceTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{

    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);
    knComputeOtherForceTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}






void manualSetting(ParticleBufferList buff_list, int nump, int step)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

    knManualSetting << <num_block, num_thread >> >(buff_list, nump, step);
}

void advance(ParticleBufferList buff_list, int nump)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

	//knIntegrateVelocitySim << <num_block, num_thread >> >(buff_list, nump);
    knIntegrateVelocityE << <num_block, num_thread >> >(buff_list, nump);


}
void advanceWave(ParticleBufferList buff_list, int nump, float time){
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);


    knIntegrateVelocitySimWave << <num_block, num_thread >> >(buff_list, nump, time);
}

void advanceMix(ParticleBufferList buff_list, int nump)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

    //CUDA_SAFE_CALL(cudaStreamWaitEvent(sms_stream, tra_force_event, 0));
    knIntegrateVelocityMix << <num_block, num_thread >> >(buff_list, nump);
}


void advancePCI(ParticleBufferList buff_list, int nump)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

    //CUDA_SAFE_CALL(cudaStreamWaitEvent(sms_stream, tra_force_event, 0));
    knIntegrateVelocity << <num_block, num_thread >> >(buff_list, nump);
}
void computeGradWValuesSimpleSMS(ParticleBufferList buff_list, int *cell_start, int *cell_end, BlockTask *block_task, int num_block, sumGrad *particle_device)
{
    if (num_block <= 0) return;

    int invocated_block = 0;
    int num_thread = kDefaultNumThreadSMS;
    int const max_block = 32768;

    while (invocated_block < num_block)
    {
        int remainder_block = num_block - invocated_block;
        int current_num_block = max_block > remainder_block ? remainder_block : max_block;
        knComputeGradWValuesSimple << <current_num_block, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_start, cell_end, block_task, invocated_block, particle_device);
        invocated_block += max_block;
    }
}

void computeGradWValuesSimpleTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_number, uint num, sumGrad *particle_device)
{
    //int total_thread = range.end - range.begin;
    if (num <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(num, num_thread);


    knComputeGradWValuesSimpleTRA << <num_block, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_number, num, particle_device);

}


void find_max_P(int blocks, int tds, sumGrad *id_value, int numbers)
{
    int iSize = 1;
    while (iSize < numbers)
    {
        iSize <<= 1;
        find_max << <blocks, tds >> >(id_value, numbers, iSize);
    }
}


void predictionCorrectionStepHybrid128(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                       float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range){
    if (nump <= 0) return;
    // int numpp = range.end - range.begin;

    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;
    //   std::cout << "asdf";
    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {


        predictPositionAndVelocity(buff_list, nump);
        max_predicted_density = 1000.0f;

        //computePredictedDensityAndPressureTRA(buff_list, cell_offset, cell_num, range, pcisph_density_factor);
        //computePredictedDensityAndPressureSMS(buff_list, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);
        computePredictedDensityAndPressureHybrid128(range, buff_list, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);
        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;


        //std::cout << "asdf";

        //computeCorrectivePressureForceTRA(buff_list, cell_offset, cell_num, range);
        //computeCorrectivePressureForce(buff_list, cell_offset, cell_num, block_task, num_block);
        computeCorrectivePressureForceHybrid128(range, buff_list, cell_offset, cell_num, block_task, num_block);

        iteration++;
    }
    //   std::cout << "asdf";
}

void predictionCorrectionStepHybrid128n(ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                        float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range){
    if (nump <= 0) return;
    // int numpp = range.end - range.begin;

    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;
    //   std::cout << "asdf";
    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {


        predictPositionAndVelocity(buff_list_n, nump);

        max_predicted_density = 1000.0f;

        //computePredictedDensityAndPressureTRA(buff_list, cell_offset, cell_num, range, pcisph_density_factor);
        //computePredictedDensityAndPressureSMS(buff_list, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);
        computePredictedDensityAndPressureHybrid128n(range, buff_list_n, cindex, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);

        //computePredictedDensityAndPressureHybrid128(range, buff_list_n,cell_offset, cell_num, block_task, num_block, pcisph_density_factor);

        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;


        //std::cout << "asdf";

        //computeCorrectivePressureForceTRA(buff_list, cell_offset, cell_num, range);
        //computeCorrectivePressureForce(buff_list, cell_offset, cell_num, block_task, num_block);
        computeCorrectivePressureForceHybrid128n(range, buff_list_n, cindex, cell_offset, cell_num, block_task, num_block);
        //computeCorrectivePressureForceHybrid128(range, buff_list_n, cell_offset, cell_num, block_task, num_block);
        iteration++;
    }
    //   std::cout << "asdf";
}


void predictionCorrectionStepHybrid(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block,
                                    float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float	pcisph_max_density_error_allowed, ParticleIdxRange range){
    if (nump <= 0) return;
    // int numpp = range.end - range.begin;

    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;
    //   std::cout << "asdf";
    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {


        predictPositionAndVelocity(buff_list, nump);
        max_predicted_density = 1000.0f;

        //computePredictedDensityAndPressureTRA(buff_list, cell_offset, cell_num, range, pcisph_density_factor);
        //computePredictedDensityAndPressureSMS(buff_list, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);
        computePredictedDensityAndPressureHybrid(range, buff_list, cell_offset, cell_num, block_task, num_block, pcisph_density_factor);
        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;


        //std::cout << "asdf";

        //computeCorrectivePressureForceTRA(buff_list, cell_offset, cell_num, range);
        //computeCorrectivePressureForce(buff_list, cell_offset, cell_num, block_task, num_block);
        computeCorrectivePressureForceHybrid(range, buff_list, cell_offset, cell_num, block_task, num_block);

        iteration++;
    }
    //   std::cout << "asdf";
}


void predictionCorrectionStepTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block
                                  , float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float pcisph_max_density_error_allowed)
{
    if (num_block <= 0) return;
    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;



    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {
        //printf("In PCISPH Loop \n");

        predictPositionAndVelocity(buff_list, nump);
        max_predicted_density = 1000.0f;

        computePredictedDensityAndPressureTRAS(buff_list, cell_offset, cell_number, block_task, num_block, pcisph_density_factor);

        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;

        computeCorrectivePressureForceTRAS(buff_list, cell_offset, cell_number, block_task, num_block);

        iteration++;
    }
    //printf("getMaxPredictedDensityCUDA outside the loop %f \n", max_predicted_density);
}

void predictionCorrectionStepSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block
                                 , float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float pcisph_max_density_error_allowed)
{
    if (num_block <= 0) return;
    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;



    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {
        //printf("In PCISPH Loop \n");

        predictPositionAndVelocity(buff_list, nump);
        max_predicted_density = 1000.0f;

        computePredictedDensityAndPressureSMS(buff_list, cell_offset, cell_number, block_task, num_block, pcisph_density_factor);

        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;

        computeCorrectivePressureForce(buff_list, cell_offset, cell_number, block_task, num_block);

        iteration++;
    }
    //printf("getMaxPredictedDensityCUDA outside the loop %f \n", max_predicted_density);
}



void predictionCorrectionStepSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block
                                   , float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float pcisph_max_density_error_allowed)
{
    if (num_block <= 0) return;
    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;



    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {
        //printf("In PCISPH Loop \n");

        predictPositionAndVelocity(buff_list, nump);
        max_predicted_density = 1000.0f;

        computePredictedDensityAndPressureSMS64(buff_list, cell_offset, cell_number, block_task, num_block, pcisph_density_factor);

        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;

        computeCorrectivePressureForce64(buff_list, cell_offset, cell_number, block_task, num_block);

        iteration++;
    }
    //printf("getMaxPredictedDensityCUDA outside the loop %f \n", max_predicted_density);
}

void predictionCorrectionStepTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num,
                                 float pcisph_density_factor, unsigned int nump, int pcisph_min_loop, int pcisph_max_loop, float pcisph_max_density_error_allowed, ParticleIdxRange range)
{
    int numpp = range.end - range.begin;
    if (numpp <= 0) return;

    bool densityErrorTooLarge = true;
    int iteration = 0;
    float max_predicted_density;



    while ((iteration < pcisph_min_loop) || ((densityErrorTooLarge) && (iteration < pcisph_max_loop)))
    {
        //printf("In PCISPH Loop \n");

        predictPositionAndVelocity(buff_list, numpp);
        max_predicted_density = 1000.0f;

        computePredictedDensityAndPressureTRA(buff_list, cell_offset, cell_num, range, pcisph_density_factor);

        //getMaxPredictedDensityCUDA(buff_list, max_predicted_density, nump);
        //printf("getMaxPredictedDensityCUDA %f \n", max_predicted_density);

        float densityErrorInPercent = max(0.1f * max_predicted_density - 100.0f, 0.0f);

        if (densityErrorInPercent < pcisph_max_density_error_allowed)
            densityErrorTooLarge = false;

        computeCorrectivePressureForceTRA(buff_list, cell_offset, cell_num, range);

        iteration++;
    }
    //printf("getMaxPredictedDensityCUDA outside the loop %f \n", max_predicted_density);
}



void predictPositionAndVelocity(ParticleBufferList buff_list, uint nump)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

    //CUDA_SAFE_CALL(cudaStreamWaitEvent(sms_stream, tra_force_event, 0));
    knPredictPositionAndVelocity << <num_block, num_thread >> >(buff_list, nump);
}

void computePredictedDensityAndPressureSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor)
{
    if (num_block <= 0) return;


    int num_thread = kDefaultNumThreadSMS;

    knComputePredictedDensityAndPressure << <num_block, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_number, block_task, pcisph_density_factor);

}

void computePredictedDensityAndPressureTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 3);

    int num_thread = 96;//kDefaultNumThreadSMS;

    knComputePredictedDensityAndPressureTRAS << <num_blocks, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_number, block_task, pcisph_density_factor);

}
void computePredictedDensityAndPressureSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor)
{
    if (num_block <= 0) return;

    int num_blocks = ceil_int(num_block, 2);
    int num_thread = 64;// kDefaultNumThreadSMS;

    knComputePredictedDensityAndPressure64 << <num_blocks, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_number, block_task, pcisph_density_factor);

}

void computePredictedDensityAndPressureHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor){
    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    knComputePredictedDensityAndPressureHybrid128 << <number_blocks, num_thread/*, 0, sms_stream*/ >> >(range, buff_list, cell_offset, cell_number, block_task, pcisph_density_factor, bt_offset);
}



void computePredictedDensityAndPressureHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor){
    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    knComputePredictedDensityAndPressureHybrid128n << <number_blocks, num_thread/*, 0, sms_stream*/ >> >(range, buff_list_n, cindex, cell_offset, cell_number, block_task, pcisph_density_factor, bt_offset);
}


void computePredictedDensityAndPressureHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task, int num_block, float pcisph_density_factor){
    int total_thread = range.end - range.begin;
    int num_thread = 32;
    int bt_offset = 0;
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, num_thread);
        num_block += bt_offset;
    }
    if (num_block <= 0) return;
    knComputePredictedDensityAndPressureHybrid << <num_block, num_thread/*, 0, sms_stream*/ >> >(range, buff_list, cell_offset, cell_number, block_task, pcisph_density_factor, bt_offset);
}


void computePredictedDensityAndPressureTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range, float pcisph_density_factor)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    knComputePredictedDensityAndPressureTRA << <num_block, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_num, range, pcisph_density_factor);//(buff_list, cell_start, cell_end, block_task, invocated_block, pcisph_density_factor);
}

void getMaxPredictedDensityCUDA(ParticleBufferList buff_list, float& max_predicted_density, unsigned int nump)
{
    float* max_predicted_density_value;
    cudaMalloc((void**)&max_predicted_density_value, sizeof(float));
    GetMaxValue << <1, 1 >> >(buff_list, max_predicted_density_value, nump);
    cudaMemcpy(&max_predicted_density, max_predicted_density_value, sizeof(float), cudaMemcpyDeviceToHost);
}

void computeCorrectivePressureForce(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;

    int num_thread = kDefaultNumThreadSMS;

    knComputeCorrectivePressureForce << <num_block, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_num, block_task);

}

void computeCorrectivePressureForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 3);
    int num_thread = 96;//kDefaultNumThreadSMS;

    knComputeCorrectivePressureForceTRAS << <num_blocks, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_num, block_task);

}

void computeCorrectivePressureForce64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 2);
    int num_thread = 64;// kDefaultNumThreadSMS;

    knComputeCorrectivePressureForce64 << <num_blocks, num_thread/*, 0, sms_stream*/ >> >(buff_list, cell_offset, cell_num, block_task);

}
void computeCorrectivePressureForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){
    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    knComputeCorrectivePressureForceHybrid128 << <number_blocks, num_thread/*, 0, sms_stream*/ >> >(range, buff_list, cell_offset, cell_num, block_task, bt_offset);
}


void computeCorrectivePressureForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){
    int total_thread = range.end - range.begin;
    int num_thread = 64;
    int bt_offset = 0;
    int number_blocks = ceil_int(num_block, 2);
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, 64);
        number_blocks += bt_offset;
    }
    if (number_blocks <= 0) return;

    knComputeCorrectivePressureForceHybrid128n << <number_blocks, num_thread/*, 0, sms_stream*/ >> >(range, buff_list_n, cindex, cell_offset, cell_num, block_task, bt_offset);
}

void computeCorrectivePressureForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block){
    int total_thread = range.end - range.begin;
    int num_thread = 32;
    int bt_offset = 0;
    if (total_thread > 0){
        bt_offset = ceil_int(total_thread, num_thread);
        num_block += bt_offset;
    }
    if (num_block <= 0) return;
    knComputeCorrectivePressureForceHybrid << <num_block, num_thread/*, 0, sms_stream*/ >> >(range, buff_list, cell_offset, cell_num, block_task, bt_offset);
}
void computeCorrectivePressureForceTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range){
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    knComputeCorrectivePressureForceTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
}




float computeDensityErrorFactorTRA(float mass, float rest_density, float time_step, ParticleBufferList buff_list, int *cell_offset, int *cell_number, uint nump)
{
    uint max_num_neighbors = 0;
    uint particle_with_max_num_neighbors = 0;

    sumGrad *particle_host;
    particle_host = (sumGrad *)malloc(sizeof(sumGrad)*nump);
    sumGrad *particle_device;
    CUDA_SAFE_CALL(cudaMalloc((void**)&particle_device, nump * sizeof(sumGrad)));


    computeGradWValuesSimpleTRA(buff_list, cell_offset, cell_number, nump, particle_device);



    cudaMemcpy(particle_host, particle_device, sizeof(sumGrad)*nump, cudaMemcpyDeviceToHost);
    printf("CUDA_SAFE_CALL(cudaMalloc((void**)&particle_device, nump * sizeof(sumGrad))): %.20f\n", particle_host[0].sumGradWDot);

    for (uint id = 0; id < nump; id++) {
        if (particle_host[id].num_neigh > max_num_neighbors) {
            max_num_neighbors = particle_host[id].num_neigh;
            particle_with_max_num_neighbors = id;
        }
    }
    float factor = computeFactorSimple(mass, rest_density, time_step, particle_with_max_num_neighbors, particle_host);
    free(particle_host);
    cudaFree(particle_device);
    return factor;
}





//pscisph over--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
}

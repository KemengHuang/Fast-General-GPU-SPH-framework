// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

namespace sph {
//sf PCISPH-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
__device__
inline int knComputeCellGradWValuesSimple(CdapSharedData *sdata, CDAPData *self_data, int read_num, sumGrad *particle_device, uint self_idx)
{
    int num = 0;



    return num;
}


__device__
inline int cdMax(int a, int b)
{
    return a>b ? a : b;
}

__global__
void find_max(sumGrad *id_value, int numbers, int iSize)
{
    int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    x_id++;
    if (x_id <= numbers)
    {
        int P = x_id & (iSize - 1);
        if (0 == P)
            P = iSize;
        if (P > (iSize >> 1))
        {
            x_id--;
            id_value[x_id].num_neigh = cdMax(id_value[x_id].num_neigh, id_value[x_id + (iSize >> 1) - P].num_neigh);
        }
    }
}





__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeGradWValuesSimple(ParticleBufferList buff_list, int *cell_start, int *cell_end, BlockTask *block_task, int bt_offset, sumGrad *particle_device)
{

}

__global__
void knPredictPositionAndVelocity(ParticleBufferList buff_list, unsigned int nump)
{

}

__device__
inline float knComputeCellDensityPredicted(pciCdapSharedData *sdata, pciCDAPData *self_data, int read_num)
{
    register float total_cell_density = 0;



    return total_cell_density;
}
__device__
inline float knComputeCellDensityPredicted9(CMDSharedData *sdata, pciCDAPData *self_data, int read_num)
{
    register float total_cell_density = 0;


    return total_cell_density;
}
__device__
inline float knComputeCellDensityPredicted9_64(CMDSharedData128 *sdata, pciCDAPData *self_data, int read_num)
{
    register float total_cell_density = 0;



    return total_cell_density;
}
__device__
inline float knComputeCellDensityPredicted128(pciCdapSharedData128 *sdata, pciCDAPData *self_data, int read_num)
{
    register float total_cell_density = 0;



    return total_cell_density;
}

__device__
float knComputeCellDensityPredictedTRA(ParticleBufferList &buff_list, pciCDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    float total_density = 0.0f;


    return total_density;
}


__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressureTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task, float pcisph_density_factor)
{

}

__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressure(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task, float pcisph_density_factor)
{

}
__global__// __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressure64(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task, float pcisph_density_factor)
{

}




__global__
__launch_bounds__(kDefaultNumThreadTRA, 8)
void knComputePredictedDensityAndPressureTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range, float pcisph_density_factor)
{

}






__device__
void knComputeCellCorrectivePressureForceTRA(float3 *pres_kn, ParticleBufferList &buff_list, pciCFData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{

}



__global__
__launch_bounds__(kDefaultNumThreadTRA, 8)
void knComputeCorrectivePressureForceTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}




__global__
void GetMaxValue(ParticleBufferList buff_list, float* max_predicted_density, unsigned int nump)
{
    uint tid = threadIdx.x;
    if (tid == 0)
    {
        float maxValue = 1000.0f;
        for (uint i = 0; i < nump; i++)
        {
            /*Particle *p = &(dMem[i]);*/
            /*if (buff_list.phase[i] != LAVA_FLUID)
            {
            continue;
            }*/
            if (buff_list.predicted_density[i]>maxValue)
            {
                maxValue = buff_list.predicted_density[i];
            }
        }
        *max_predicted_density = maxValue;
    }
}

__device__
inline void knComputeCellCorrectivePressureForce(float3 *pres_kn, CfkSharedData *sdata, pciCFData *self_data, int read_num)
{
    //#pragma unroll 16

}
__device__
inline void knComputeCellCorrectivePressureForce9(float3 *pres_kn, CKSharedData *sdata, pciCFData *self_data, int read_num)
{

}
__device__
inline void knComputeCellCorrectivePressureForce9_64(float3 *pres_kn, CKSharedData128 *sdata, pciCFData *self_data, int read_num)
{

}
__device__
inline void knComputeCellCorrectivePressureForce128(float3 *pres_kn, CfkSharedData128 *sdata, pciCFData *self_data, int read_num)
{


}
__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForce(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task)
{

}

__global__// __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task)
{

}
__global__// __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForce64(ParticleBufferList buff_list, int *cell_offset, int *cell_nump, BlockTask *block_task)
{

}

__device__
float knComputeCellGradWValuesSimpleTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_number, ushort3 cell_pos, sumGrad *particle_device, int &self_idx)
{
    float total_density = 0.0f;


    return total_density;
}

__global__
__launch_bounds__(kDefaultNumThreadTRA, 8)
void knComputeGradWValuesSimpleTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_number, uint num, sumGrad *particle_device)
{

}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressureHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, float pcisph_density_factor, int bt_offset)
{

}

__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressureHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, float pcisph_density_factor, int bt_offset)
{

}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressureHybrid128(ParticleIdxRange range, ParticleBufferList buff_list_n, ParticleBufferList buff_list_o, int *cell_offset, int *cell_num, BlockTask *block_task, float pcisph_density_factor, int bt_offset)
{

}

__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputePredictedDensityAndPressureHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, float pcisph_density_factor, int bt_offset)
{

}

__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

}

__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

}

__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeCorrectivePressureForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

}
}

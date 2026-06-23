// Auto-generated declarations for integration_kernels.cu.

#ifndef _SOLVER_INTEGRATION_KERNELS_CUH_
#define _SOLVER_INTEGRATION_KERNELS_CUH_

#include "solver/kernel_common.cuh"

namespace sph {

__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knInit(ParticleBufferList buff_list, int nump);
__global__
void knManualSetting(ParticleBufferList buff_list, unsigned int nump, int step);
__global__
void knIntegrateVelocityMix(ParticleBufferList buff_list, unsigned int nump);
__device__
inline float4 addfloat4(const float4& vec4, const float3 &vec3);
__device__
inline float4 floathalf4add3(float3 ra, const float4& vec4);
__device__
inline float3 float4m3(float3 ra, const float4& vec4);
__device__ 
inline float dotV34(float3 v3, float4 v4);
__global__
void knIntegrateVelocitySimWave(ParticleBufferList buff_list, unsigned int nump, float time);
__global__
void knIntegrateVelocitySim(ParticleBufferList buff_list, unsigned int nump);
__global__
void knIntegrateVelocityE(ParticleBufferList buff_list, unsigned int nump);
__global__
void knIntegrateVelocity(ParticleBufferList buff_list, unsigned int nump);

}

#endif

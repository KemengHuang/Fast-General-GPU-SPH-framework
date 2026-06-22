//
// sph_kernel.cu
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#include "sph_kernel.cuh"
#include "sph_kernel_shared_data.cuh"

namespace sph
{

const int kDefaultNumThreadSMS = 32;
const int kDefulatMinBlocksSMS = 10;
const int kDefaultNumThreadSMS2 = 64;
const int kDefaultNumThreadTRA = 128;
const int kNumberNeighborCells = 27;

__constant__ SystemParameter kDevSysPara;

__device__ __host__
inline float3 cal_rePos(const float4 &pos4, const float4 &pos3) {
    return make_float3(pos3.x - pos4.x, pos3.y - pos4.y, pos3.z - pos4.z);
}


__device__ __host__
inline unsigned int ceil_uint(unsigned int a, unsigned int b) { return (a + b - 1) / b; }

__device__ __host__
inline float powf_2(float base) { return base * base; }

__device__ __host__
inline float powf_3(float base) { return base * base * base; }

__device__ __host__
inline float powf_7(float base) { return base * base * base * base * base * base * base; }

typedef unsigned int uint;

#define COLORA(r,g,b,a)	( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

/****************************** Kernel ******************************/

//sf 计算density 一个cell

__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knInit(ParticleBufferList buff_list, int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    if (idx >= nump) return;
    float4 ev;
    ev.x = 0; ev.y = 0; ev.z = 0; ev.w = 0;
    buff_list.evaluated_velocity[idx] = ev;
    //    buff_list.density[idx] = 0.f;
    buff_list.acceleration[idx] = make_float3(0.f, 0.f, 0.f);
    //    buff_list.pressure[idx] = 0.f;
}


void BuffInit(ParticleBufferList buff_list_n, int nm){
    if (nm <= 0) return;
    int num_thread = 256;
    int number_block = ceil_int(nm, num_thread);
    knInit << <number_block, num_thread >> >(buff_list_n, nm);
}

__device__
inline void knComputeCellDensitySMS64(const int& isSame, SimDenSharedData128 *sdata, CDAPData *self_data, int read_num)
{
    //   register float total_cell_density = 0;
    int kk = (1-isSame)*(threadIdx.x>>5);
    for (size_t i = (kk <<5); i < (kk<<5) + read_num; ++i)
    {
        float4 neighbor_position = sdata->getPos(i);
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        self_data->pos.w += powf_3(kDevSysPara.kernel_2 - dis_2);
    }

    //    return total_cell_density;
}
__device__
inline void knComputeCellDensitySMS(SimDenSharedData *sdata, CDAPData *self_data, int read_num)
{
    register float total_cell_density = 0;
    //  int kk = threadIdx.x / 32;
    for (size_t i = 0; i < read_num; ++i)
    {
        float4 neighbor_position = sdata->getPos(i);
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        self_data->pos.w += powf_3(kDevSysPara.kernel_2 - dis_2);
    }

    //   return total_cell_density;
}
__device__
inline void knComputeCellForceSMS64(const int& isSame, float3 *pres_kn, float3 *vis_kn, SimForSharedData128 *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16
	int kk = (1 - isSame)*(threadIdx.x>>5);
    //float vis_kn;
    for (size_t i = (kk <<5); i < (kk <<5) + read_num; ++i)
    {

        float4 neighbor_position = sdata->getPosition(i);
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);

        //          self_data->pos - neighbor_position;

        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;

        float dis = sqrtf(dis_2);
        float V = 1 / neighbor_position.w;
        float kernel_r = kDevSysPara.kernel - dis;

        // pressure force
        float temp_pres_kn = V * (self_data->ev.w + sdata->getEV(i).w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * __fdividef(temp_pres_kn, dis);

        // viscosity force
        float3 rel_vel = cal_rePos(self_data->ev, sdata->getEV(i));// sdata->getEV(i) - self_data->ev;
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        // surface force
        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 3 / 4 * (kDevSysPara.kernel_2 - dis_2));
    }
}
__device__
inline void knComputeCellForceSMS(float3 *pres_kn, float3 *vis_kn, SimForSharedData *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16

    //float vis_kn;
    for (size_t i = 0; i < read_num; ++i)
    {
        float4 neighbor_position = sdata->getPosition(i);
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);
        //           self_data->pos - neighbor_position;

        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;

        float dis = sqrtf(dis_2);
        float V = 1 / neighbor_position.w;
        float kernel_r = kDevSysPara.kernel - dis;

        // pressure force
        float temp_pres_kn = V * (self_data->ev.w + sdata->getEV(i).w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * __fdividef(temp_pres_kn, dis);

        // viscosity force
        float3 rel_vel = cal_rePos(self_data->ev, sdata->getEV(i));// sdata->getEV(i) - self_data->ev;
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        // surface force
        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 3 / 4 * (kDevSysPara.kernel_2 - dis_2));
    }
}
//sf 计算density
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeDensitySMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{
    BlockTask bt = block_task[blockIdx.x];

	int cell_id;// = CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);

    register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

    register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];


    register float total_density = 0.0f;
    register CDAPData data;

    if (self_idx < temp_cell_end)   // initialize self data
    {
        data.pos = buff_list.position_d[self_idx];
    }
    __shared__ SimDenSharedData sdata;
//    sdata.initialize(cell_offset, cell_num, bt.cell_pos, kDevSysPara.grid_size);
    while (true)
    {
        int r = sdata.read32Data(buff_list);

        if (0 == r) break;  // neighbor cells read complete
        __syncthreads();
        if (self_idx < temp_cell_end)
        {
            knComputeCellDensitySMS(&sdata, &data, r);
        }
    }
    if (self_idx < temp_cell_end)
    {
        data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
        data.pos.w += kDevSysPara.self_density;
        buff_list.position_d[self_idx].w = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
        buff_list.pressure[self_idx] = (powf_7(__fdividef(data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;
    }
}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeDensitySMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{
    int t = blockIdx.x;// -bt_offset;
    int n = 2 * t + threadIdx.x / 32;
    BlockTask bt = block_task[n];
	int isSame = bt.isSame;
	int cell_id;// = CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);

    register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x % 32; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

    register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];


    register float total_density = 0.0f;
    register CDAPData data;

    if (self_idx < temp_cell_end)   // initialize self data
    {
        data.pos = tex1Dfetch(texRef, self_idx);
    //    data.pos =buff_list.position_d[self_idx];
        data.pos.w = 0;
    }
    __shared__ SimDenSharedData128 sdata;
//    sdata.initialize(bt.xxi, bt.xxx, nullptr, isSame, cell_offset, cell_num, bt.cell_pos, kDevSysPara.grid_size);
    while (true)
    {
		int r;// = sdata.read32Data(isSame, buff_list);

        if (0 == r) break;  // neighbor cells read complete
        __syncthreads();
        if (self_idx < temp_cell_end)
        {
            knComputeCellDensitySMS64(isSame, &sdata, &data, r);
        }
    }
    if (self_idx < temp_cell_end)
    {
        data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
        data.pos.w += kDevSysPara.self_density;
        buff_list.position_d[self_idx].w = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
        buff_list.evaluated_velocity[self_idx].w = (powf_7(__fdividef(data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;
    }
}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{
    BlockTask bt = block_task[blockIdx.x];

	int cell_id;// = CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);

    register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

    register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];

    register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
    register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);
    register CFData self_data;

    if (self_idx < temp_cell_end)   // init self data
    {
        self_data.pos = buff_list.position_d[self_idx];
        self_data.ev = buff_list.evaluated_velocity[self_idx];
        //        self_data.pres = buff_list.pressure[self_idx];
        self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
        self_data.lplc_color = 0.0f;
    }


    __shared__ SimForSharedData sdata;
  //  sdata.initialize(cell_offset, cell_num, bt.cell_pos, kDevSysPara.grid_size);
    while (true)
    {
        int r = sdata.read32Data(buff_list);

        if (0 == r) break;  // neighbor cells read complete
        __syncthreads();
        if (self_idx < temp_cell_end)
        {
            knComputeCellForceSMS(&pres_kn, &vis_kn, &sdata, &self_data, r);
        }
    }
    if (self_idx < temp_cell_end)
    {
        register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

        self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
        self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

        self_data.lplc_color = __fdividef(self_data.lplc_color, buff_list.position_d[self_idx].w);
        float sur_nor = sqrtf(self_data.grad_color.x * self_data.grad_color.x +
                              self_data.grad_color.y * self_data.grad_color.y +
                              self_data.grad_color.z * self_data.grad_color.z);
        // buff_list.surface_normal_vector[self_idx] = sur_nor;

        float3 force;
        //force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        if (sur_nor > kDevSysPara.surface_normal)
        {
            force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        }
        else
        {
            force = make_float3(0.0f, 0.0f, 0.0f);
        }

        total_force *= kDevSysPara.mass;
        buff_list.acceleration[self_idx] = total_force + force;
    }
}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{
    int t = blockIdx.x;// -bt_offset;
    int n = 2 * t + threadIdx.x / 32;
    BlockTask bt = block_task[n];
	int isSame = bt.isSame;
	int cell_id;// = CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);

    register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x % 32; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

    register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];

    register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
    register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);
    register CFData self_data;

    if (self_idx < temp_cell_end)   // init self data
    {
        self_data.pos = tex1Dfetch(texRef, self_idx);
     //   self_data.pos = buff_list.position_d[self_idx];
        self_data.ev = tex1Dfetch(texRefe, self_idx);
   //     self_data.ev = buff_list.evaluated_velocity[self_idx];

        self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
        self_data.lplc_color = 0.0f;
    }


    __shared__ SimForSharedData128 sdata;
//	sdata.initialize(bt.xxi, bt.xxx, nullptr, isSame, cell_offset, cell_num, bt.cell_pos, kDevSysPara.grid_size);
    while (true)
    {
		int r;// = sdata.read32Data(isSame, buff_list);

        if (0 == r) break;  // neighbor cells read complete
        __syncthreads();
        if (self_idx < temp_cell_end)
        {
            knComputeCellForceSMS64(isSame, &pres_kn, &vis_kn, &sdata, &self_data, r);
        }
    }
    if (self_idx < temp_cell_end)
    {
        register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

        self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
        self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

        self_data.lplc_color = __fdividef(self_data.lplc_color, buff_list.position_d[self_idx].w);
        float sur_nor = sqrtf(self_data.grad_color.x * self_data.grad_color.x +
                              self_data.grad_color.y * self_data.grad_color.y +
                              self_data.grad_color.z * self_data.grad_color.z);
        // buff_list.surface_normal_vector[self_idx] = sur_nor;

        float3 force;
        //force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        if (sur_nor > kDevSysPara.surface_normal)
        {
            force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        }
        else
        {
            force = make_float3(0.0f, 0.0f, 0.0f);
        }

        total_force *= kDevSysPara.mass;
        buff_list.acceleration[self_idx] = total_force + force;
    }
}
//sf 计算 force 一个cell
__device__
inline void knComputeCellOtherForceSMS(float3 *boundary_force, float3 *vis_kn, pmfCdapSharedData *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16

}
__device__
inline void knComputeCellOtherForceSMS9(float3 *boundary_force, float3 *vis_kn, CMFSharedData *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16

}
__device__
inline void knComputeCellOtherForceSMS9_64(float3 *boundary_force, float3 *vis_kn, CMFSharedData128 *sdata, CFData *self_data, int read_num)
{


}

__device__
inline void knComputeCellOtherForceSMS128(float3 *boundary_force, float3 *vis_kn, pmfCdapSharedData128 *sdata, CFData *self_data, int read_num)
{

}
//sf 计算force
__global__ __launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeOtherForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task)
{

}
__global__ //__launch_bounds__(kDefaultNumThreadSMS, kDefulatMinBlocksSMS)
void knComputeOtherForceSMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_number, BlockTask *block_task)
{


}
__device__
void knComputeCellOtherForceTRA(float3 *vis_kn, ParticleBufferList &buff_list, CFData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{

}

__global__
__launch_bounds__(kDefaultNumThreadTRA, 8)
void knComputeOtherForceTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}






__global__
void knManualSetting(ParticleBufferList buff_list, unsigned int nump, int step)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

    if (idx >= nump) return;
}

//sf 更新速度位置 新的处理方式
//__global__
//void knIntegrateVelocity(ParticleBufferList buff_list, unsigned int nump)
//{
//    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
//
//    if (idx >= nump) return;
//
//    float3 position = buff_list.position[idx];
//    float3 velocity = buff_list.velocity[idx];
//    float3 eval_vel = buff_list.evaluated_velocity[idx];
//    float3 accelerate = buff_list.acceleration[idx];
//    float diff, speed;
//
//    const float epsilon = 0.00001f;
//    const float alpha = 20000000;
//    const float beta = 200000;
//
//    // Y-axis
//    diff = kDevSysPara.kernel - (position.y - kDevSysPara.bound_min.y);
//    if (diff > 0.0f)
//    {
//        accelerate.y += alpha * diff - beta * eval_vel.y;
//    }
//    diff = kDevSysPara.kernel - (kDevSysPara.bound_max.y - position.y);
//    if (diff > 0.0f)
//    {
//        accelerate.y -= alpha * diff + beta * eval_vel.y;
//    }
//    // X-axis
//    diff = kDevSysPara.kernel - (position.x - kDevSysPara.bound_min.x);
//    if (diff > 0.0f)
//    {
//        accelerate.x += alpha * diff - beta * eval_vel.x;
//    }
//    diff = kDevSysPara.kernel - (kDevSysPara.bound_max.x - position.x);
//    if (diff > 0.0f)
//    {
//        accelerate.x -= alpha * diff + beta * eval_vel.x;
//    }
//    // Z-axis
//    diff = kDevSysPara.kernel - (position.z - kDevSysPara.bound_min.z);
//    if (diff > 0.0f)
//    {
//        accelerate.z += alpha * diff - beta * eval_vel.z;
//    }
//    diff = kDevSysPara.kernel - (kDevSysPara.bound_max.z - position.z);
//    if (diff > 0.0f)
//    {
//        accelerate.z -= alpha * diff + beta * eval_vel.z;
//    }
//
//    const float acc_limit = 10000;
//    const float vel_limit = 25;
//
//    // acceleration limit
//    speed = accelerate.x * accelerate.x + accelerate.y * accelerate.y + accelerate.z * accelerate.z;
//    if (speed > acc_limit)
//    {
//        accelerate;
//    }
//
//    velocity += (accelerate / buff_list.density[idx] + kDevSysPara.gravity) * kDevSysPara.time_step;
//    position += velocity * kDevSysPara.time_step;
//
//    //if (idx == 32)
//    //{
//    //    printf("idx: %d, vel: %f, %f, %f\n", idx, velocity.x, velocity.y, velocity.z);
//    //}
//
//    //buff_list.color[idx] = COLORA(0.2 + 0.8 * position.x / kDevSysPara.world_size.x,
//    //                              0.2 + 0.5 * position.y / kDevSysPara.world_size.y,
//    //                              0.2 + 0.8 * position.z / kDevSysPara.world_size.z, 1);
//	//buff_list.color[idx] = COLORA(0.2,
//	//                              0.2 + 0.2 * position.y / kDevSysPara.world_size.y,
//	//							  0.4 + 0.3 * position.z / kDevSysPara.world_size.z + 0.3 * position.x / kDevSysPara.world_size.x, 1);
//    buff_list.position[idx] = position;
//    buff_list.velocity[idx] = velocity;
//    buff_list.evaluated_velocity[idx] = (buff_list.evaluated_velocity[idx] + velocity) / 2;
//    buff_list.final_position[idx] = position * kDevSysPara.sim_ratio + kDevSysPara.sim_origin;  // sf final_position似乎是画在屏幕上的位置
//}

__global__
void knIntegrateVelocityMix(ParticleBufferList buff_list, unsigned int nump)
{

}
__device__
inline float4 addfloat4(const float4& vec4, const float3 &vec3){
    //  make_float3()
    float4 nVec4;
    nVec4.x = vec4.x + vec3.x;
    nVec4.y = vec4.y + vec3.y;
    nVec4.z = vec4.z + vec3.z;
    nVec4.w = vec4.w;
    return nVec4;
}

__device__
inline float4 floathalf4add3(float3 ra, const float4& vec4){
    float4 v4;
    v4.x = (ra.x + vec4.x)*0.5;;
    v4.y = (ra.y + vec4.y)*0.5;
    v4.z = (vec4.z + ra.z)*0.5;
    v4.w = vec4.w;
    return v4;
}
__device__
inline float3 float4m3(float3 ra, const float4& vec4){

    return make_float3(ra.x*vec4.x, ra.y*vec4.y, vec4.z*ra.z);
}
__device__ 
inline float dotV34(float3 v3, float4 v4){
    return v3.x*v4.x + v3.y*v4.y + v3.z*v4.z;
}
__global__
void knIntegrateVelocitySimWave(ParticleBufferList buff_list, unsigned int nump, float time)
{
    unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

    if (idx >= nump) return;

    register float4 position = tex1Dfetch(texRef, idx);
    register float4 eval_vel = tex1Dfetch(texRefe, idx);
    float3 t_velocity = buff_list.velocity[idx];
    float3 accelerate = buff_list.acceleration[idx];// / t_position.w + kDevSysPara.gravity;
    float diff, speed;
    float3 norm;
    float adj;
    const float epsilon = 0.00001f;
    const float alpha = 20000000;
    const float beta = 200000;
    float slop = 0.09;

    register float kernel = kDevSysPara.kernel + 0.01;

    diff = kernel - (position.y - (kDevSysPara.bound_min.y + (position.x - kDevSysPara.bound_min.x)*slop));
    if (diff > 0.0f)
    {
        norm = make_float3(-slop, 1.0 - slop, 0);
        adj = alpha * diff - beta * dotV34(norm, eval_vel);
        norm *= adj; accelerate += norm;
    }

    diff = kernel - (kDevSysPara.bound_max.y - position.y);
    if (diff > 0.0f)
    {
        accelerate.y -= alpha * diff + beta * eval_vel.y;
    }

    float seq = 3;
    float rd = sin(time * seq);
    diff = kernel - (position.x - (kDevSysPara.bound_min.x + 1.2*(0.5*rd + 0.5)));
    if (diff > 0.0f)
    {
        accelerate.x += (alpha * diff*pow(5000 * (position.y - kDevSysPara.bound_min.y) / (kDevSysPara.bound_max.y - kDevSysPara.bound_min.y), 2) - beta * eval_vel.x);
    }


    diff = kernel - (kDevSysPara.bound_max.x - position.x);
    if (diff > 0.0f)
    {
        accelerate.x -= alpha * diff + beta * eval_vel.x;
    }

    diff = kernel - (position.z - kDevSysPara.bound_min.z);
    if (diff > 0.0f)
    {
        accelerate.z += alpha * diff - beta * eval_vel.z;
    }
    diff = kernel - (kDevSysPara.bound_max.z - position.z);
    if (diff > 0.0f)
    {
        accelerate.z -= alpha * diff + beta * eval_vel.z;
    }

    const float acc_limit = 3000000;
    const float vel_limit = 36;
    accelerate = (accelerate / position.w + kDevSysPara.gravity);
    speed = accelerate.x * accelerate.x + accelerate.y * accelerate.y + accelerate.z * accelerate.z;
    if (speed > acc_limit)
        accelerate *= 1 / sqrtf(acc_limit);
    t_velocity += accelerate * kDevSysPara.time_step;
    speed = t_velocity.x*t_velocity.x + t_velocity.y*t_velocity.y + t_velocity.z*t_velocity.z;
    if (speed > vel_limit)
        t_velocity *= 1 / sqrtf(vel_limit);

    position = addfloat4(position, t_velocity * kDevSysPara.time_step);

    if (position.x >= kDevSysPara.bound_max.x){
        t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
        position.x = kDevSysPara.bound_max.x - kernel;
    }
    if (position.x <= kDevSysPara.bound_min.x){
        t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
        position.x = kDevSysPara.bound_min.x + kernel;
    }
    if (position.y >= kDevSysPara.bound_max.y){
        t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
        position.y = kDevSysPara.bound_max.y - kernel;
    }
    if (position.y <= kDevSysPara.bound_min.y + (position.x - kDevSysPara.bound_min.x)*slop){
        t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
        position.y = kDevSysPara.bound_min.y + kernel + (position.x - kDevSysPara.bound_min.x)*slop;
    }
    if (position.z >= kDevSysPara.bound_max.z){
        t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
        position.z = kDevSysPara.bound_max.z - kernel;
    }
    if (position.z <= kDevSysPara.bound_min.z){
        t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
        position.z = kDevSysPara.bound_min.z + kernel;
    }

    float denv = (4000 - position.w) / 6000;
    buff_list.color[idx] = COLORA(0.8f * denv, 0.7f * denv + 0.1, 0.8, 1.0);

    buff_list.position_d[idx] = position;
    buff_list.velocity[idx] = t_velocity;
    buff_list.evaluated_velocity[idx] = floathalf4add3(t_velocity, buff_list.evaluated_velocity[idx]);
    buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}
__global__
void knIntegrateVelocitySim(ParticleBufferList buff_list, unsigned int nump)
{
	unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

	if (idx >= nump) return;

	register float4 t_position = tex1Dfetch(texRef, idx);
	register float4 eval_vel = tex1Dfetch(texRefe, idx);

	//       register float4 t_position = buff_list.position_d[idx];
	//       register float4 eval_vel = buff_list.evaluated_velocity[idx];

	float3 t_velocity = buff_list.velocity[idx];
	float3 acc_temp = buff_list.acceleration[idx] / t_position.w + kDevSysPara.gravity;
	float diff, speed;


	float length = sqrt(acc_temp.x*acc_temp.x + acc_temp.y * acc_temp.y + acc_temp.z * acc_temp.z);
	if (length > (kDevSysPara.kernel / (5 * kDevSysPara.time_step* kDevSysPara.time_step)))
		acc_temp = acc_temp*(kDevSysPara.kernel / (5 * kDevSysPara.time_step* kDevSysPara.time_step)) / length;

	float3 velocity = t_velocity + (acc_temp)* kDevSysPara.time_step;
	float4 position = addfloat4(t_position, velocity * kDevSysPara.time_step);// t_position + velocity * kDevSysPara.time_step;

	const float BOUNDARY = 0.0001f;

	if (position.x >= kDevSysPara.world_size.x - BOUNDARY)
	{
		velocity.x = velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.world_size.x - BOUNDARY;
	}

	if (position.x < 0.0f + BOUNDARY)
	{
		velocity.x = velocity.x * kDevSysPara.wall_damping;
		position.x = 0.0f + BOUNDARY;
	}

	if (position.y >= kDevSysPara.world_size.y - BOUNDARY)
	{
		velocity.y = velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.world_size.y - BOUNDARY;
	}

	if (position.y < 0.0f + BOUNDARY)
	{
		velocity.y = velocity.y * kDevSysPara.wall_damping;
		position.y = 0.0f + BOUNDARY;
	}

	if (position.z >= kDevSysPara.world_size.z - BOUNDARY)
	{
		velocity.z = velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.world_size.z - BOUNDARY;
	}

	if (position.z < 0.0f + BOUNDARY)
	{
		velocity.z = velocity.z * kDevSysPara.wall_damping;
		position.z = 0.0f + BOUNDARY;
	}

	buff_list.position_d[idx] = position;
	buff_list.velocity[idx] = velocity;
	buff_list.evaluated_velocity[idx] = floathalf4add3(velocity, buff_list.evaluated_velocity[idx]);
	buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}
//sf 更新位置速度  大的固体边界，原先的边界处理方式

__global__
void knIntegrateVelocityE(ParticleBufferList buff_list, unsigned int nump)
{
	unsigned int idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);

	if (idx >= nump) return;

	register float4 t_position = tex1Dfetch(texRef, idx);
	register float4 eval_vel = tex1Dfetch(texRefe, idx);

	//       register float4 t_position = buff_list.position_d[idx];
	//       register float4 eval_vel = buff_list.evaluated_velocity[idx];


	float3 t_velocity = buff_list.velocity[idx];
	float3 accelerate = buff_list.acceleration[idx];

	float diff, speed;
	float3 norm;
	float adj;
	const float alpha = 20000000;
	const float beta = 200000;

	register float kernel = kDevSysPara.kernel + 0.01;

	diff = kernel - (t_position.y - kDevSysPara.bound_min.y);
	if (diff > 0.0f)
	{
		accelerate.y += alpha * diff - beta * t_velocity.y;
	}
	diff = kernel - (kDevSysPara.bound_max.y - t_position.y);
	if (diff > 0.0f)
	{
		accelerate.y -= alpha * diff + beta * t_velocity.y;
	}

	diff = kernel - (t_position.x - kDevSysPara.bound_min.x);
	if (diff > 0.0f)
	{
		accelerate.x += alpha * diff - beta * t_velocity.x;
	}
	diff = kernel - (kDevSysPara.bound_max.x - t_position.x);
	if (diff > 0.0f)
	{
		accelerate.x -= alpha * diff + beta * t_velocity.x;
	}
	// Z-axis
	diff = kernel - (t_position.z - kDevSysPara.bound_min.z);
	if (diff > 0.0f)
	{
		accelerate.z += alpha * diff - beta * t_velocity.z;
	}
	diff = kernel - (kDevSysPara.bound_max.z - t_position.z);
	if (diff > 0.0f)
	{
		accelerate.z -= alpha * diff + beta * t_velocity.z;
	}
	const float acc_limit = 3000000;
	const float vel_limit = 36;
	accelerate = (accelerate / t_position.w + kDevSysPara.gravity);
	speed = accelerate.x * accelerate.x + accelerate.y * accelerate.y + accelerate.z * accelerate.z;
	if (speed > acc_limit)
		accelerate *= 1 / sqrtf(acc_limit);
	t_velocity += accelerate * kDevSysPara.time_step;

	speed = t_velocity.x*t_velocity.x + t_velocity.y*t_velocity.y + t_velocity.z*t_velocity.z;
	if (speed > vel_limit)
		t_velocity *= 1 / sqrtf(vel_limit);
	float4 position = addfloat4(t_position, t_velocity * kDevSysPara.time_step);

	if (position.x >= kDevSysPara.bound_max.x){
		t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.bound_max.x - kernel;
	}
	if (position.x <= kDevSysPara.bound_min.x){
		t_velocity.x = t_velocity.x * kDevSysPara.wall_damping;
		position.x = kDevSysPara.bound_min.x + kernel;
	}
	if (position.y >= kDevSysPara.bound_max.y){
		t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.bound_max.y - kernel;
	}
	if (position.y <= kDevSysPara.bound_min.y){
		t_velocity.y = t_velocity.y * kDevSysPara.wall_damping;
		position.y = kDevSysPara.bound_min.y + kernel;
	}
	if (position.z >= kDevSysPara.bound_max.z){
		t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.bound_max.z - kernel;
	}
	if (position.z <= kDevSysPara.bound_min.z){
		t_velocity.z = t_velocity.z * kDevSysPara.wall_damping;
		position.z = kDevSysPara.bound_min.z + kernel;
	}

	buff_list.position_d[idx] = position;
	buff_list.velocity[idx] = t_velocity;
	buff_list.evaluated_velocity[idx] = floathalf4add3(t_velocity, buff_list.evaluated_velocity[idx]);
	buff_list.final_position[idx] = float4m3(kDevSysPara.sim_ratio, position) + kDevSysPara.sim_origin;
}
__global__
void knIntegrateVelocity(ParticleBufferList buff_list, unsigned int nump)
{

}

//sf PCISPH-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//sf 计算PCISPH密度误差因子
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

//sf step1 预测位置
__global__
void knPredictPositionAndVelocity(ParticleBufferList buff_list, unsigned int nump)
{

}

//sf step2 计算预测后的密度
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




//sf step3 找最大密度误差
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

//sf step4 计算pressure产生的力
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
//sf 计算force
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

//sf PCISPH over -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


/****************************** Interface ******************************/

cudaStream_t sms_stream;
cudaEvent_t sms_density_event;
cudaEvent_t sms_force_event;

void transSysParaToDevice(const SystemParameter *host_para)
{
    CUDA_SAFE_CALL(cudaMemcpyToSymbol(kDevSysPara, host_para, sizeof(SystemParameter)));
    printf("testing self_density %f\n", kDevSysPara.mass);
}

void initializeKernel()
{
    CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&sms_stream, cudaStreamDefault));
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&sms_density_event, cudaEventDisableTiming));
    CUDA_SAFE_CALL(cudaEventCreateWithFlags(&sms_force_event, cudaEventDisableTiming));
}

void releaseKernel()
{
    CUDA_SAFE_CALL(cudaStreamDestroy(sms_stream));
    CUDA_SAFE_CALL(cudaEventDestroy(sms_density_event));
    CUDA_SAFE_CALL(cudaEventDestroy(sms_force_event));
}

//sf host计算density的总函数
void computeDensitySMS64(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_blocks = ceil_int(num_block, 2);
    int num_thread = 64;
    cudaCreateTextureObject(&texRef,
        &(cudaResourceDesc){
        .resType = cudaResourceTypeLinear,
            .res.linear = { ptr, cudaCreateChannelDesc<float4>(), 100 * sizeof(float4) }
    },
        & (cudaTextureDesc) {
        .readMode = cudaReadModeElementType
    },
        NULL);
    //cudaBindTexture(0, texRef, buff_list.position_d);
    cudaCreateTextureObject(&texRef, &(cudaResourceDesc){cudaResourceTypeLinear, { .linear = {ptr,cudaCreateChannelDesc<float4>(),100 * sizeof(float4)} }}, & (cudaTextureDesc) { .readMode = cudaReadModeElementType }, NULL);
    knComputeDensitySMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_num, block_task);

    cudaUnbindTexture(texRef);

}

void computeDensitySMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_thread = 32;

    knComputeDensitySMS << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, block_task);

}
__device__
void knComputeCellDensityTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    //    float total_density = 0.0f;

    int cell_id = CellPos2CellIdx(cell_pos, kDevSysPara.grid_size);
    if (kInvalidCellIdx == cell_id) return;

    int start_idx = cell_offset[cell_id];
    int end_idx = start_idx + cell_num[cell_id];

    if (0xffffffff == start_idx) return;

    for (size_t i = start_idx; i < end_idx; ++i)
    {
        float4 neighbor_pos = tex1Dfetch(texRef, i);
        //      float4 neighbor_pos = buff_list.position_d[i];


        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);// self_data->pos - neighbor_pos;
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        self_data->pos.w += __powf(kDevSysPara.kernel_2 - dis_2, 3);
    }

    //    return total_density;
}
__device__
void knComputeCellDensityTRA9(ParticleBufferList &buff_list, CDAPData *self_data, int cell_offset, int cell_num)
{
    if (0 == cell_num) return;
    int end_idx = cell_offset + cell_num;
    for (size_t i = cell_offset; i < end_idx; ++i)
    {
        float4 neighbor_pos = tex1Dfetch(texRef, i);
      //  float4 neighbor_pos = buff_list.position_d[i];


        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);// self_data->pos - neighbor_pos;
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        self_data->pos.w += __powf(kDevSysPara.kernel_2 - dis_2, 3);
    }

    //    return total_density;
}
__device__
float knComputeCellMixDensityTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    float total_density = 0.0f;

    int cell_id = CellPos2CellIdx(cell_pos, kDevSysPara.grid_size);
    if (kInvalidCellIdx == cell_id) return 0.0f;

    int start_idx = cell_offset[cell_id];
    int end_idx = start_idx + cell_num[cell_id];

    if (0xffffffff == start_idx) return 0.0f;

    for (size_t i = start_idx; i < end_idx; ++i)
    {
        float4 neighbor_pos = buff_list.position_d[i];

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        total_density += __powf(kDevSysPara.kernel_2 - dis_2, 3)*(kDevSysPara.mass1*buff_list.vlfrt[i].a1 + kDevSysPara.mass2*buff_list.vlfrt[i].a2);
    }

    return total_density;
}
__global__
void knComputeDensityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{
    int self_idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x) + range.begin;

    if (self_idx >= range.end) return;

    register CDAPData self_data;
    self_data.pos = tex1Dfetch(texRef, self_idx);
    //    self_data.pos = buff_list.position_d[self_idx];

    self_data.pos.w = 0;


    ushort3 cell_pos = ParticlePos2CellPos(self_data.pos, kDevSysPara.cell_size);

    register ushort3 grid_size = kDevSysPara.grid_size;
    register int cell_offset_;
    register int cell_nump_;

    for (int i = 0; i < 9; i++){
        ushort3 neighbor_pos = cell_pos + make_ushort3(-1, i % 3 - 1, i / 3 % 3 - 1);
        if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
            neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
            cell_offset_ = 0;
            cell_nump_ = 0;
        }
        else {
            int nid_left, nid_mid, nid_right;
            nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
            ++neighbor_pos.x;
            nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
            ++neighbor_pos.x;
            nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
            cell_offset_ =
                kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
            int my_cell_nump = cell_num[nid_mid];
            if (kInvalidCellIdx != nid_left) my_cell_nump += cell_num[nid_left];
            if (kInvalidCellIdx != nid_right) my_cell_nump += cell_num[nid_right];
            cell_nump_ = my_cell_nump;
            knComputeCellDensityTRA9(buff_list, &self_data, cell_offset_, cell_nump_);
        }
    }

    



    //for (int z = -1; z <= 1; ++z)
    //{
    //for (int y = -1; y <= 1; ++y)
    //{
    //for (int x = -1; x <= 1; ++x)
    //{
    //ushort3 neigbor_cell_pos = cell_pos + make_ushort3(x, y, z);
    //knComputeCellDensityTRA(buff_list, &self_data, cell_offset, cell_num, neigbor_cell_pos);
    //}
    //}
    //}

    self_data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
    self_data.pos.w += kDevSysPara.self_density;
    buff_list.position_d[self_idx].w = self_data.pos.w;
    buff_list.evaluated_velocity[self_idx].w = (__powf(__fdividef(self_data.pos.w, kDevSysPara.rest_density), 7) - 1) * kDevSysPara.gas_constant;
}
__global__
void knComputeMixDensityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}
void computeDensityTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    cudaBindTexture(0, texRef, buff_list.position_d);
    knComputeDensityTRA << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, range);
    cudaUnbindTexture(texRef);
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
__device__
void knComputeCellForceTRA(float3 *pres_kn, float3 *vis_kn, ParticleBufferList &buff_list, CFData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    //float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

    int cell_id = CellPos2CellIdx(cell_pos, kDevSysPara.grid_size);
    if (kInvalidCellIdx == cell_id) return;// total_force;

    int start_idx = cell_offset[cell_id];
    int end_idx = start_idx + cell_num[cell_id];
    if (0xffffffff == start_idx) return;// total_force;

    for (size_t i = start_idx; i < end_idx; ++i)
    {
             register float4 neighbor_pos = tex1Dfetch(texRef, i);
             register float4 neighbor_ev = tex1Dfetch(texRefe, i);

   //     register float4 neighbor_pos = buff_list.position_d[i];
   //     register float4 neighbor_ev = buff_list.evaluated_velocity[i];

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        float dis = sqrtf(dis_2);
        float V = 1 / (neighbor_pos.w);
        float kernel_r = kDevSysPara.kernel - dis;

        // pressure force
        float temp_pres_kn = V * (self_data->ev.w + neighbor_ev.w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * __fdividef(temp_pres_kn, dis);
        //float pressure_kernel = kDevSysPara.spiky_value * kernel_r * kernel_r;
        //float temp_force = V * (self_data->pres + buff_list.pressure[i]) * pressure_kernel;
        //total_force -= rel_pos * __fdividef(temp_force, dis);

        // viscosity force
        float3 rel_vel = cal_rePos(self_data->ev, neighbor_ev);//buff_list.evaluated_velocity[i] - self_data->ev;
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;
        //float3 rel_vel = buff_list.evaluated_velocity[i] - self_data->ev;
        //float viscosity_kernel = kDevSysPara.visco_value * kernel_r;
        //float temp_force2 = V * kDevSysPara.viscosity * viscosity_kernel;
        //total_force += rel_vel * temp_force2;

        // surface force
        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 3 / 4 * (kDevSysPara.kernel_2 - dis_2));
    }

    // return total_force;
}
__device__
void knComputeCellForceTRA9(float3 *pres_kn, float3 *vis_kn, ParticleBufferList &buff_list, CFData *self_data, int cell_offset, int cell_num)
{

    if (0 == cell_num) return;
    int end_idx = cell_offset + cell_num;
    for (size_t i = cell_offset; i < end_idx; ++i)
    {
             register float4 neighbor_pos = tex1Dfetch(texRef, i);
             register float4 neighbor_ev = tex1Dfetch(texRefe, i);

   //     register float4 neighbor_pos = buff_list.position_d[i];
   //     register float4 neighbor_ev = buff_list.evaluated_velocity[i];

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        float dis = sqrtf(dis_2);
        float V = 1 / (neighbor_pos.w);
        float kernel_r = kDevSysPara.kernel - dis;


        float temp_pres_kn = V * (self_data->ev.w + neighbor_ev.w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * __fdividef(temp_pres_kn, dis);

        float3 rel_vel = cal_rePos(self_data->ev, neighbor_ev);//buff_list.evaluated_velocity[i] - self_data->ev;
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 3 / 4 * (kDevSysPara.kernel_2 - dis_2));
    }

    // return total_force;
}
__device__
void knComputeCellGradDataTRA(ParticleBufferList &buff_list, GrediData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{

}


__device__
void knComputeVFracTRA(ParticleBufferList &buff_list, DivergData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{

}
__device__
void knComputeAccTRA(ParticleBufferList &buff_list, DivergTenData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{

}


__global__
void knComputeForceTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{
    int self_idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x) + range.begin;

    if (self_idx >= range.end) return;

    register CFData self_data;
    self_data.pos = tex1Dfetch(texRef, self_idx);
    self_data.ev = tex1Dfetch(texRefe, self_idx);

    //    self_data.pos = buff_list.position_d[self_idx];
    //    self_data.ev = buff_list.evaluated_velocity[self_idx];


    self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
    self_data.lplc_color = 0.0f;


    ushort3 cell_pos = ParticlePos2CellPos(self_data.pos, kDevSysPara.cell_size);

    //float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
    register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);






    register ushort3 grid_size = kDevSysPara.grid_size;
    register int cell_offset_;
    register int cell_nump_;

    for (int i = 0; i < 9; i++){
        ushort3 neighbor_pos = cell_pos + make_ushort3(-1, i % 3 - 1, i / 3 % 3 - 1);
        if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
            neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
            continue;
        }
        else {
            int nid_left, nid_mid, nid_right;
            nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
            ++neighbor_pos.x;
            nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
            ++neighbor_pos.x;
            nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
            cell_offset_ =
                kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
            cell_nump_ = cell_num[nid_mid];
            if (kInvalidCellIdx != nid_left) cell_nump_ += cell_num[nid_left];
            if (kInvalidCellIdx != nid_right) cell_nump_ += cell_num[nid_right];
            knComputeCellForceTRA9(&pres_kn, &vis_kn, buff_list, &self_data, cell_offset_, cell_nump_);
        }
    }

    






    //for (int z = -1; z <= 1; ++z)
    //{
    //for (int y = -1; y <= 1; ++y)
    //{
    //for (int x = -1; x <= 1; ++x)
    //{
    //ushort3 neigbor_cell_pos = cell_pos + make_ushort3(x, y, z);
    //knComputeCellForceTRA(&pres_kn, &vis_kn, buff_list, &self_data, cell_offset, cell_num, neigbor_cell_pos);
    //}
    //}
    //}

    register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

    self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
    self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

    self_data.lplc_color = __fdividef(self_data.lplc_color, self_data.pos.w);
    float sur_nor = sqrtf(self_data.grad_color.x * self_data.grad_color.x +
                          self_data.grad_color.y * self_data.grad_color.y +
                          self_data.grad_color.z * self_data.grad_color.z);
    //buff_list.surface_normal_vector[self_idx] = sur_nor;

    float3 force;
    //force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
    if (sur_nor > kDevSysPara.surface_normal)
    {
        force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
    }
    else
    {
        force = make_float3(0.0f, 0.0f, 0.0f);
    }

    total_force *= kDevSysPara.mass;// / buff_list.density[self_idx];
    buff_list.acceleration[self_idx] = total_force + force;
}

__global__
void knComputeDriftVelocityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}

__global__
void kncomputeVolumeFracTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}
__global__
void kncomputeAccelTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{


}
void computeForceTRA(ParticleBufferList buff_list, ParticleIdxRange range, int *cell_offset, int *cell_num)
{
    int total_thread = range.end - range.begin;
    if (total_thread <= 0) return;

    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(total_thread, num_thread);

    cudaBindTexture(0, texRef, buff_list.position_d);
    cudaBindTexture(0, texRefe, buff_list.evaluated_velocity);
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
    cudaBindTexture(0, texRef, buff_list.position_d);
    cudaBindTexture(0, texRefe, buff_list.evaluated_velocity);
  
    knComputeForceSMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_num, block_task);
  
}

void computeForceSMS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int num_block)
{
    if (num_block <= 0) return;
    int num_thread = 32;

    //knComputeOtherForceSMS64 << <num_blocks, num_thread >> >(buff_list, cell_offset, cell_number, block_task);
    knComputeForceSMS << <num_block, num_thread >> >(buff_list, cell_offset, cell_num, block_task);

}

__global__ //__launch_bounds__(128, kDefulatMinBlocksSMS)
void knComputeOtherForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{

}
//sf host计算force的总函数
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


__global__ __launch_bounds__(128, kDefulatMinBlocksSMS)
void knComputeOtherForceHybrid(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

}




__global__ //__launch_bounds__(128, kDefulatMinBlocksSMS)
void knComputeOtherForceHybrid128(ParticleIdxRange range, ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

}

__global__ //__launch_bounds__(128, kDefulatMinBlocksSMS)
void knComputeOtherForceHybrid128n(ParticleIdxRange range, ParticleBufferList buff_list_n, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{

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
__device__
ushort3 calCI(const ushort3& in){
	ushort3 outd;
	outd.x = in.x >> 2;
	outd.y = in.y >> 2;
	outd.z = in.z >> 2;
	return outd;
}
__global__ //__launch_bounds__(64, 10)
void kncomputeDensityHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{
    if (blockIdx.x < bt_offset){
        int self_idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x) + range.begin;
        if (self_idx >= range.end) return;
        self_idx = cindex[self_idx];

        register CDAPData self_data;
        self_data.pos = tex1Dfetch(texRef, self_idx);
  //      self_data.pos =buff_list.position_d[self_idx];
        self_data.pos.w = 0;
        ushort3 cell_posc = ParticlePos2CellPosM(self_data.pos, kDevSysPara.cell_size);

		ushort3 cell_pos = calCI(cell_posc);
		int xxx = (cell_posc.x) & 0x03;

        register ushort3 grid_size = kDevSysPara.grid_size;
        register int cell_offset_;
        register int cell_nump_;

        for (int i = 0; i < 9; i++){
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, i % 3 - 1, i / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                continue;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
              /*  cell_offset_ =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                cell_nump_ = cell_num[nid_mid];
                if (kInvalidCellIdx != nid_left) cell_nump_ += cell_num[nid_left];
                if (kInvalidCellIdx != nid_right) cell_nump_ += cell_num[nid_right];*/








				cell_offset_ =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset_M[(nid_left << 6) + (xxx << 4)];
				cell_nump_ = cell_num[nid_mid];
				if (kInvalidCellIdx != nid_left) cell_nump_ += cell_offset[nid_mid] - cell_offset_M[(nid_left << 6) + (xxx << 4)];//cell_nump[nid_left];
				if (xxx == 3){
					if (kInvalidCellIdx != nid_right) cell_nump_ += cell_num[nid_right];
				}
				else{
					if (kInvalidCellIdx != nid_right) cell_nump_ += cell_offset_M[(nid_right << 6) + ((xxx + 1) << 4)] - cell_offset_M[(nid_right << 6)];//cell_nump[nid_right];
				}
				//cell_nump_[kk] = my_cell_nump;








                knComputeCellDensityTRA9(buff_list, &self_data, cell_offset_, cell_nump_);
            }
        }

        /*        for (int z = -1; z <= 1; ++z)
        {
        for (int y = -1; y <= 1; ++y)
        {
        for (int x = -1; x <= 1; ++x)
        {
        ushort3 neigbor_cell_pos = cell_pos + make_ushort3(x, y, z);
        knComputeCellDensityTRA(buff_list, &self_data, cell_offset, cell_num, neigbor_cell_pos);
        }
        }
        }*/

        self_data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
        self_data.pos.w += kDevSysPara.self_density;
        buff_list.position_d[self_idx].w = self_data.pos.w;
        buff_list.evaluated_velocity[self_idx].w = (__powf(__fdividef(self_data.pos.w, kDevSysPara.rest_density), 7) - 1) * kDevSysPara.gas_constant;

		float denv = (5000 - buff_list.position_d[self_idx].w) / 6000;
		buff_list.color[self_idx] = COLORA(1.0f*denv, 0.f, 1.0*denv, 1.0);
    }
    else{

        int t = blockIdx.x - bt_offset;
		int n = (t << 1) + (threadIdx.x >> 5);
        BlockTask bt = block_task[n];
		int isSame = bt.isSame;
	
        int cell_id = bt.cellid;
		ushort3 cellpos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);
		/*char a = bt.yyi;
		char b = bt.yyy;
		char c = bt.zzi;
		char d = bt.zzz;*/

        register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x % 32; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

        register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];

        register float total_density = 0.0f;
        register CDAPData data;

        if (self_idx < temp_cell_end)   // initialize self data
        {
            data.pos = tex1Dfetch(texRef, self_idx);
        //    data.pos = buff_list.position_d[self_idx];
            data.pos.w = 0;
        }
        __shared__ SimDenSharedData128 sdata;
		sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
        while (true)
        {
			__syncthreads();
			int r = sdata.read32Data(cell_offset_M, isSame, buff_list);
			__syncthreads();
		//	if (r > 32 && r < 64 && (threadIdx.x == 0 || threadIdx.x == 32)) printf("%d\n", r);
            if (0 == r) break;  // neighbor cells read complete
      //      __syncthreads();
            if (self_idx < temp_cell_end)
            {
				knComputeCellDensitySMS64(isSame, &sdata, &data, r);
            }
        }
        if (self_idx < temp_cell_end)
        {
            data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
            data.pos.w += kDevSysPara.self_density;
            buff_list.position_d[self_idx].w = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
            buff_list.evaluated_velocity[self_idx].w = (powf_7(__fdividef(data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;

			float denv = (5000 - buff_list.position_d[self_idx].w) / 6000;
			if (isSame == 1){
				buff_list.color[self_idx] = COLORA(0.f, 1.0f*denv, 1.0*denv, 1.0);
			}
			else{
				buff_list.color[self_idx] = COLORA(1.0f*denv, 1.0f*denv, 0.f, 1.0);
			}


        }
    }
}
__global__ //__launch_bounds__(64, 10)
void kncomputeForceHybrid128n(int *cell_offset_M,ParticleIdxRange range, ParticleBufferList buff_list, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, int bt_offset)
{
    if (blockIdx.x < bt_offset){
        int self_idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x) + range.begin;
        if (self_idx >= range.end) return;
        self_idx = cindex[self_idx];

        register CFData self_data;
        self_data.pos = tex1Dfetch(texRef, self_idx);
     //   self_data.pos = buff_list.position_d[self_idx];
        self_data.ev = tex1Dfetch(texRefe, self_idx); 
   //     self_data.ev = buff_list.evaluated_velocity[self_idx];
       

        self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
        self_data.lplc_color = 0.0f;


		ushort3 cell_posc = ParticlePos2CellPosM(self_data.pos, kDevSysPara.cell_size);

		ushort3 cell_pos = calCI(cell_posc);
		int xxx = (cell_posc.x) & 0x03;
        //float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
        register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
        register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);




        register ushort3 grid_size = kDevSysPara.grid_size;
        register int cell_offset_;
        register int cell_nump_;

        for (int i = 0; i < 9; i++){
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, i % 3 - 1, i / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                continue;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                /*cell_offset_ =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                cell_nump_ = cell_num[nid_mid];
                if (kInvalidCellIdx != nid_left) cell_nump_ += cell_num[nid_left];
                if (kInvalidCellIdx != nid_right) cell_nump_ += cell_num[nid_right];*/

				cell_offset_ =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset_M[(nid_left << 6) + (xxx << 4)];
				cell_nump_ = cell_num[nid_mid];
				if (kInvalidCellIdx != nid_left) cell_nump_ += cell_offset[nid_mid] - cell_offset_M[(nid_left << 6) + (xxx << 4)];//cell_nump[nid_left];
				if (xxx == 3){
					if (kInvalidCellIdx != nid_right) cell_nump_ += cell_num[nid_right];
				}
				else{
					if (kInvalidCellIdx != nid_right) cell_nump_ += cell_offset_M[(nid_right << 6) + ((xxx + 1) << 4)] - cell_offset_M[(nid_right << 6)];//cell_nump[nid_right];
				}

                knComputeCellForceTRA9(&pres_kn, &vis_kn, buff_list, &self_data, cell_offset_, cell_nump_);
            }
        }

        /* for (int z = -1; z <= 1; ++z)
        {
        for (int y = -1; y <= 1; ++y)
        {
        for (int x = -1; x <= 1; ++x)
        {
        ushort3 neigbor_cell_pos = cell_pos + make_ushort3(x, y, z);
        knComputeCellForceTRA(&pres_kn, &vis_kn, buff_list, &self_data, cell_offset, cell_num, neigbor_cell_pos);
        }
        }
        }*/

        register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

        self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
        self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

        self_data.lplc_color = __fdividef(self_data.lplc_color, self_data.pos.w);
        float sur_nor = sqrtf(self_data.grad_color.x * self_data.grad_color.x +
                              self_data.grad_color.y * self_data.grad_color.y +
                              self_data.grad_color.z * self_data.grad_color.z);
        //buff_list.surface_normal_vector[self_idx] = sur_nor;

        float3 force;
        //force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        if (sur_nor > kDevSysPara.surface_normal)
        {
            force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
        }
        else
        {
            force = make_float3(0.0f, 0.0f, 0.0f);
        }

        total_force *= kDevSysPara.mass;// / buff_list.density[self_idx];
        buff_list.acceleration[self_idx] = total_force + force;

		
		//buff_list.color
    }
    else{

        int t = blockIdx.x - bt_offset;
		int n = (t<<1) + (threadIdx.x>>5);
		BlockTask bt = block_task[n];
		int isSame = bt.isSame;

		int cell_id = bt.cellid;// CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);
		ushort3 cellpos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);

	/*	char a = bt.yyi;
		char b = bt.yyy;
		char c = bt.zzi;
		char d = bt.zzz;*/

        register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x % 32; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

        register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];

        register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
        register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);
        register CFData self_data;

        if (self_idx < temp_cell_end)   // init self data
        {
			
            self_data.pos = tex1Dfetch(texRef, self_idx);
       //     self_data.pos = buff_list.position_d[self_idx];
            self_data.ev = tex1Dfetch(texRefe, self_idx);
        //    self_data.ev = buff_list.evaluated_velocity[self_idx];
            self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
            self_data.lplc_color = 0.0f;
        }

        __shared__ SimForSharedData128 sdata;
		sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
        while (true)
        {
			__syncthreads();
			int r = sdata.read32Data(cell_offset_M, isSame, buff_list);
			__syncthreads();
            if (0 == r) break;  // neighbor cells read complete
            
            if (self_idx < temp_cell_end)
            {
                knComputeCellForceSMS64(isSame, &pres_kn, &vis_kn, &sdata, &self_data, r);
            }
        }
        if (self_idx < temp_cell_end)
        {
            register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

            self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
            self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

            self_data.lplc_color = __fdividef(self_data.lplc_color, self_data.pos.w);
            float sur_nor = sqrtf(self_data.grad_color.x * self_data.grad_color.x +
                                  self_data.grad_color.y * self_data.grad_color.y +
                                  self_data.grad_color.z * self_data.grad_color.z);
            // buff_list.surface_normal_vector[self_idx] = sur_nor;

            float3 force;
            //force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
            if (sur_nor > kDevSysPara.surface_normal)
            {
                force = self_data.grad_color * kDevSysPara.surface_coe * self_data.lplc_color / sur_nor;
            }
            else
            {
                force = make_float3(0.0f, 0.0f, 0.0f);
            }

            total_force *= kDevSysPara.mass;
            buff_list.acceleration[self_idx] = total_force + force;
        }
    }
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
    cudaBindTexture(0, texRef, buff_list_n.position_d);
//	std::cout << ceil_int(num_block, 2) << "               " << num_block << "         asfasdfasfasdfafsd" << std::endl;
	kncomputeDensityHybrid128n << <number_blocks, num_thread >> >(cell_offset_M, range, buff_list_n, cindex, cell_offset, cell_num, block_task, bt_offset);
    cudaUnbindTexture(texRef);
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
    cudaBindTexture(0, texRef, buff_list_n.position_d);
    cudaBindTexture(0, texRefe, buff_list_n.evaluated_velocity);
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

//sf 更新位置速度
void advance(ParticleBufferList buff_list, int nump)
{
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);

	//knIntegrateVelocitySim << <num_block, num_thread >> >(buff_list, nump);
    knIntegrateVelocityE << <num_block, num_thread >> >(buff_list, nump);
    cudaUnbindTexture(texRef);
    cudaUnbindTexture(texRefe);


}
void advanceWave(ParticleBufferList buff_list, int nump, float time){
    int num_thread = kDefaultNumThreadTRA;
    int num_block = ceil_int(nump, num_thread);


    knIntegrateVelocitySimWave << <num_block, num_thread >> >(buff_list, nump, time);
    cudaUnbindTexture(texRef);
    cudaUnbindTexture(texRefe);
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
//sf host计算force的总函数
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


//sf PCISPH预测校验主函数----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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


        predictPositionAndVelocity(buff_list, nump);   //sf 有固液交互步骤
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


        predictPositionAndVelocity(buff_list_n, nump);   //sf 有固液交互步骤

        //predictPositionAndVelocity(buff_list_o, nump);   //sf 有固液交互步骤
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


        predictPositionAndVelocity(buff_list, nump);   //sf 有固液交互步骤
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

        //      predictPositionAndVelocity(num_block, buff_list, nump, cell_start, cell_end, block_task, num_block);   //sf 有固液交互步骤
        predictPositionAndVelocity(buff_list, nump);   //sf 有固液交互步骤
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

        //      predictPositionAndVelocity(num_block, buff_list, nump, cell_start, cell_end, block_task, num_block);   //sf 有固液交互步骤
        predictPositionAndVelocity(buff_list, nump);   //sf 有固液交互步骤
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

        //      predictPositionAndVelocity(num_block, buff_list, nump, cell_start, cell_end, block_task, num_block);   //sf 有固液交互步骤
        predictPositionAndVelocity(buff_list, nump);   //sf 有固液交互步骤
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

        //      predictPositionAndVelocity(num_block, buff_list, nump, cell_start, cell_end, block_task, num_block);   //sf 有固液交互步骤
        predictPositionAndVelocity(buff_list, numpp);   //sf 有固液交互步骤
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



    cudaMemcpy(particle_host, particle_device, sizeof(sumGrad)*nump, cudaMemcpyDeviceToHost);//传回主机端
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


//sf finally namespace sph-------------------------------------------------
}
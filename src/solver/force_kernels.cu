// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

// Switch for the force SMS different-cell path:
// 0 = original shared-memory path for both same-cell and different-cell tasks
// 1 = register-load + warp-local shared exchange for different-cell tasks,
//     shared-memory path kept for same-cell tasks (dominant case)
#ifndef FORCE_SMS_USE_REGISTER_PATH
#define FORCE_SMS_USE_REGISTER_PATH 1
#endif

namespace sph {
__device__
inline void knComputeCellForceSMS64(const int& isSame, float3 *pres_kn, float3 *vis_kn, SimForSharedData128 *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16
	int kk = (1 - isSame)*(threadIdx.x>>5);
    //float vis_kn;
    for (int i = (kk <<5); i < (kk <<5) + read_num; ++i)
    {

        float4 neighbor_position = sdata->getPosition(i);
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);

        //          self_data->pos - neighbor_position;

        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;

        float inv_dis = rsqrtf(dis_2);
        float dis = dis_2 * inv_dis;
        float V = neighbor_position.w;  // position_d.w stores 1/density
        float kernel_r = kDevSysPara.kernel - dis;

        // pressure force
        float temp_pres_kn = V * (self_data->ev.w + sdata->getEV(i).w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * (temp_pres_kn * inv_dis);

        // viscosity force
        float3 rel_vel = cal_rePos(self_data->ev, sdata->getEV(i));// sdata->getEV(i) - self_data->ev;
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        // surface force
        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 0.75f * (kDevSysPara.kernel_2 - dis_2));
    }
}
__device__
inline void knComputeCellForceSMS(float3 *pres_kn, float3 *vis_kn, SimForSharedData *sdata, CFData *self_data, int read_num)
{
    //#pragma unroll 16

    //float vis_kn;
    for (int i = 0; i < read_num; ++i)
    {
        float4 neighbor_position = sdata->getPosition(i);
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);
        //           self_data->pos - neighbor_position;

        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;

        float dis = sqrtf(dis_2);
        float V = neighbor_position.w;  // position_d.w stores 1/density
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
            (dis_2 - 0.75f * (kDevSysPara.kernel_2 - dis_2));
    }
}
// WARNING: dead/broken kernel kept for reference only — cell_id and the shared-memory
// staging calls are commented out, so launching it is undefined behavior.
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
        self_data.pos = __ldg(&buff_list.position_d[self_idx]);
        self_data.ev = __ldg(&buff_list.evaluated_velocity[self_idx]);
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

        self_data.lplc_color = self_data.lplc_color * buff_list.position_d[self_idx].w;  // position_d.w stores 1/density
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
// WARNING: dead/broken kernel kept for reference only — cell_id and the shared-memory
// staging calls are commented out, so launching it is undefined behavior.
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
        self_data.pos = buff_list.position_d[self_idx];
     //   self_data.pos = buff_list.position_d[self_idx];
        self_data.ev = buff_list.evaluated_velocity[self_idx];
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

        self_data.lplc_color = self_data.lplc_color * buff_list.position_d[self_idx].w;  // position_d.w stores 1/density
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

#if FORCE_SMS_USE_REGISTER_PATH
__device__ __forceinline__
void knComputeCellForceReg64(float4 *shared_pos, float4 *shared_ev, int warp_base,
                             float3 *pres_kn, float3 *vis_kn,
                             CFData *self_data, int read_num)
{
    // Ensure the loop bound is uniform across the warp.
    read_num = __shfl_sync(0xFFFFFFFF, read_num, 0);
    for (int i = warp_base; i < warp_base + read_num; ++i)
    {
        float4 neighbor_position = shared_pos[i];
        float4 neighbor_ev = shared_ev[i];
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);

        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;

        float inv_dis = rsqrtf(dis_2);
        float dis = dis_2 * inv_dis;
        float V = neighbor_position.w;  // position_d.w stores 1/density
        float kernel_r = kDevSysPara.kernel - dis;

        // pressure force
        float temp_pres_kn = V * (self_data->ev.w + neighbor_ev.w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * (temp_pres_kn * inv_dis);

        // viscosity force
        float3 rel_vel = cal_rePos(self_data->ev, neighbor_ev);
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        // surface force
        float temp = V * powf_2(kDevSysPara.kernel_2 - dis_2);
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * (kDevSysPara.kernel_2 - dis_2) *
            (dis_2 - 0.75f * (kDevSysPara.kernel_2 - dis_2));
    }
}
#endif

__device__
inline void knComputeCellOtherForceSMS128(float3 *boundary_force, float3 *vis_kn, pmfCdapSharedData128 *sdata, CFData *self_data, int read_num)
{

}
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






__device__
void knComputeCellForceTRA(float3 *pres_kn, float3 *vis_kn, ParticleBufferList &buff_list, CFData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    //float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

    int cell_id = CellPos2CellIdx(cell_pos, kDevSysPara.grid_size);
    if (kInvalidCellIdx == cell_id) return;// total_force;

    int start_idx = cell_offset[cell_id];
    int end_idx = start_idx + cell_num[cell_id];
    if (0xffffffff == start_idx) return;// total_force;

    for (int i = start_idx; i < end_idx; ++i)
    {
             register float4 neighbor_pos = __ldg(&buff_list.position_d[i]);
             register float4 neighbor_ev = __ldg(&buff_list.evaluated_velocity[i]);

   //     register float4 neighbor_pos = buff_list.position_d[i];
   //     register float4 neighbor_ev = buff_list.evaluated_velocity[i];

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        float dis = sqrtf(dis_2);
        float V = (neighbor_pos.w);  // position_d.w stores 1/density
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
            (dis_2 - 0.75f * (kDevSysPara.kernel_2 - dis_2));
    }

    // return total_force;
}
__device__
void knComputeCellForceTRA9(float3 *pres_kn, float3 *vis_kn, ParticleBufferList &buff_list, CFData *self_data, int cell_offset, int cell_num)
{

    if (0 == cell_num) return;
    int end_idx = cell_offset + cell_num;
    for (int i = cell_offset; i < end_idx; ++i)
    {
        float4 neighbor_pos = __ldg(&buff_list.position_d[i]);
        float4 neighbor_ev  = __ldg(&buff_list.evaluated_velocity[i]);

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        float inv_dis = rsqrtf(dis_2);
        float dis = dis_2 * inv_dis;
        float V = (neighbor_pos.w);  // position_d.w stores 1/density
        float kernel_r = kDevSysPara.kernel - dis;

        float temp_pres_kn = V * (self_data->ev.w + neighbor_ev.w) * kernel_r * kernel_r;
        *pres_kn -= rel_pos * (temp_pres_kn * inv_dis);

        float3 rel_vel = cal_rePos(self_data->ev, neighbor_ev);
        float temp_vis_kn = V * kernel_r;
        *vis_kn += rel_vel * temp_vis_kn;

        float h2_r2 = kDevSysPara.kernel_2 - dis_2;
        float temp = V * h2_r2 * h2_r2;
        self_data->grad_color += rel_pos * temp;
        self_data->lplc_color += V * h2_r2 *
            (dis_2 - 0.75f * h2_r2);
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
    self_data.pos = buff_list.position_d[self_idx];
    self_data.ev = buff_list.evaluated_velocity[self_idx];

    //    self_data.pos = buff_list.position_d[self_idx];
    //    self_data.ev = buff_list.evaluated_velocity[self_idx];


    self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
    self_data.lplc_color = 0.0f;


    ushort3 cell_pos = ParticlePos2CellPos(self_data.pos, kDevSysPara.inv_cell_size);

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

    self_data.lplc_color = self_data.lplc_color * self_data.pos.w;  // pos.w holds 1/density (loaded from position_d)
    float sur_nor_sq = self_data.grad_color.x * self_data.grad_color.x +
                       self_data.grad_color.y * self_data.grad_color.y +
                       self_data.grad_color.z * self_data.grad_color.z;
    float inv_sur_nor = rsqrtf(sur_nor_sq);
    float sur_nor = sur_nor_sq * inv_sur_nor;
    //buff_list.surface_normal_vector[self_idx] = sur_nor;

    float3 force;
    if (sur_nor > kDevSysPara.surface_normal)
    {
        force = self_data.grad_color * (kDevSysPara.surface_coe * self_data.lplc_color * inv_sur_nor);
    }
    else
    {
        force = make_float3(0.0f, 0.0f, 0.0f);
    }

    total_force *= kDevSysPara.mass;
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
__global__ //__launch_bounds__(128, kDefulatMinBlocksSMS)
void knComputeOtherForceTRAS(ParticleBufferList buff_list, int *cell_offset, int *cell_num, BlockTask *block_task)
{

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
__global__ __launch_bounds__(64, 10)
void kncomputeForceHybrid128n(int *cell_offset_M,ParticleIdxRange range, ParticleBufferList buff_list, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, const int *d_num_block, const int *d_middle)
{
    // Device-side TRA/SMS split: the host over-provisions the grid and excess
    // blocks exit immediately, so the frame needs no host readback/synchronization.
    int middle = __ldg(d_middle);
    if (middle < 0 || middle > range.end) middle = range.end;
    const int bt_offset = (middle - range.begin + 63) >> 6;  // ceil((middle - begin) / 64)

    if (blockIdx.x < bt_offset){
        int self_idx = threadIdx.x + __umul24(blockIdx.x, blockDim.x) + range.begin;
        if (self_idx >= middle) return;
        self_idx = __ldg(&cindex[self_idx]);

        register CFData self_data;
        self_data.pos = __ldg(&buff_list.position_d[self_idx]);
        self_data.ev = __ldg(&buff_list.evaluated_velocity[self_idx]); 
   //     self_data.ev = buff_list.evaluated_velocity[self_idx];
       

        self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
        self_data.lplc_color = 0.0f;


		ushort3 cell_posc = ParticlePos2CellPosM(self_data.pos, kDevSysPara.inv_cell_size);

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
					kInvalidCellIdx == nid_left ? __ldg(&cell_offset[nid_mid]) : __ldg(&cell_offset_M[(nid_left << 6) + (xxx << 4)]);
				cell_nump_ = __ldg(&cell_num[nid_mid]);
				if (kInvalidCellIdx != nid_left) cell_nump_ += __ldg(&cell_offset[nid_mid]) - __ldg(&cell_offset_M[(nid_left << 6) + (xxx << 4)]);
				if (xxx == 3){
					if (kInvalidCellIdx != nid_right) cell_nump_ += __ldg(&cell_num[nid_right]);
				}
				else{
					if (kInvalidCellIdx != nid_right) cell_nump_ += __ldg(&cell_offset_M[(nid_right << 6) + ((xxx + 1) << 4)]) - __ldg(&cell_offset_M[(nid_right << 6)]);
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

        self_data.lplc_color = self_data.lplc_color * self_data.pos.w;  // pos.w holds 1/density (loaded from position_d)
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
        if ((t << 1) >= __ldg(d_num_block)) return;  // over-provisioned SMS block (block-uniform exit)
		int n = (t<<1) + (threadIdx.x>>5);
		BlockTask bt = block_task[n];
        int isSame = 0;// bt.isSame;

		int cell_id = bt.cellid;// CellPos2CellIdx(bt.cell_pos, kDevSysPara.grid_size);
		ushort3 cellpos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);

	/*	char a = bt.yyi;
		char b = bt.yyy;
		char c = bt.zzi;
		char d = bt.zzz;*/

        register int cell_off = __ldg(&cell_offset[cell_id]);
        register int cell_np = __ldg(&cell_num[cell_id]);
        register int self_idx = cell_off + bt.p_offset + threadIdx.x % 32; //__mul24(bt.sub_idx, blockDim.x) + threadIdx.x;

        register int temp_cell_end = cell_off + cell_np;

        register float3 pres_kn = make_float3(0.0f, 0.0f, 0.0f);
        register float3 vis_kn = make_float3(0.0f, 0.0f, 0.0f);
        register CFData self_data;

        bool active = (self_idx < temp_cell_end);
        if (active)   // init self data
        {
            self_data.pos = __ldg(&buff_list.position_d[self_idx]);
            self_data.ev = __ldg(&buff_list.evaluated_velocity[self_idx]);
            self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
            self_data.lplc_color = 0.0f;
        }

#if FORCE_SMS_USE_REGISTER_PATH
        // Register-load + warp-local shared exchange for different-cell tasks.
        // (The isSame==1 shared-memory variant was removed: isSame is hard-wired to 0.)
        {
            bool warp_has_work = (bt.p_offset < cell_np);
            __shared__ SimForRegData128 sdata;
            __shared__ float4 shared_pos[64];
            __shared__ float4 shared_ev[64];
            sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);

            if (warp_has_work)
            {
                while (true)
                {
                    float4 my_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    float4 my_ev = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    int r = sdata.read32DataReg(cell_offset_M, isSame, buff_list, my_pos, my_ev);
                    if (0 == r) break;

                    shared_pos[threadIdx.x] = my_pos;
                    shared_ev[threadIdx.x] = my_ev;
                    __syncwarp();

                    // Call the helper from every lane in the warp.  It uses
                    // __shfl_sync internally, so all 32 lanes must participate.
                    int warp_base = (threadIdx.x >> 5) << 5;
                    knComputeCellForceReg64(shared_pos, shared_ev, warp_base,
                                            &pres_kn, &vis_kn, &self_data, r);
                    __syncwarp();
                }
            }
        }
#else
        // Original shared-memory SMS path for both same-cell and different-cell tasks.
        __shared__ SimForSharedData128 sdata;
        sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
        while (true)
        {
            __syncthreads();
            int r = sdata.read32Data(cell_offset_M, isSame, buff_list);
            __syncthreads();
            if (0 == r) break;
            if (active)
            {
                knComputeCellForceSMS64(isSame, &pres_kn, &vis_kn, &sdata, &self_data, r);
            }
        }
#endif

        if (self_idx < temp_cell_end)
        {
            register float3 total_force = pres_kn * kDevSysPara.spiky_value / 2 + vis_kn * kDevSysPara.viscosity * kDevSysPara.visco_value;

            self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
            self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

            self_data.lplc_color = self_data.lplc_color * self_data.pos.w;  // pos.w holds 1/density (loaded from position_d)
            float sur_nor_sq = self_data.grad_color.x * self_data.grad_color.x +
                               self_data.grad_color.y * self_data.grad_color.y +
                               self_data.grad_color.z * self_data.grad_color.z;
            float inv_sur_nor = rsqrtf(sur_nor_sq);
            float sur_nor = sur_nor_sq * inv_sur_nor;
            // buff_list.surface_normal_vector[self_idx] = sur_nor;

            float3 force;
            if (sur_nor > kDevSysPara.surface_normal)
            {
                force = self_data.grad_color * (kDevSysPara.surface_coe * self_data.lplc_color * inv_sur_nor);
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
}

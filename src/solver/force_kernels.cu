// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

// Select the live SMS implementation:
// 0 = legacy shared-memory iterator and neighbor staging
// 1 = register iterator state with warp-local neighbor staging
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

#if GSPH_USE_REGISTER_SMS
__device__ __forceinline__
void accumulateForceRegisterBatch(const float4 *neighbor_positions,
                                  const float4 *neighbor_velocities,
                                  int warp_base, float3 *pressure_sum,
                                  float3 *viscosity_sum, CFData *self_data,
                                  int neighbor_count)
{
    for (int i = warp_base; i < warp_base + neighbor_count; ++i)
    {
        float4 neighbor_position = neighbor_positions[i];
        float3 rel_pos = cal_rePos(neighbor_position, self_data->pos);

        float distance_squared = rel_pos.x * rel_pos.x
            + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (kDevSysPara.kernel_2 < distance_squared ||
            kFloatSmall > distance_squared)
            continue;

        float4 neighbor_velocity = neighbor_velocities[i];
        float inverse_distance = rsqrtf(distance_squared);
        float distance = distance_squared * inverse_distance;
        float inverse_density = neighbor_position.w;
        float kernel_distance = kDevSysPara.kernel - distance;
        float weighted_kernel_distance = inverse_density * kernel_distance;

        // pressure force
        float pressure_weight = weighted_kernel_distance
            * (self_data->ev.w + neighbor_velocity.w) * kernel_distance;
        *pressure_sum -= rel_pos * (pressure_weight * inverse_distance);

        // viscosity force
        float3 relative_velocity = cal_rePos(self_data->ev, neighbor_velocity);
        *viscosity_sum += relative_velocity * weighted_kernel_distance;

        // surface force
        float h2_r2 = kDevSysPara.kernel_2 - distance_squared;
        float weighted_h2 = inverse_density * h2_r2;
        float gradient_weight = weighted_h2 * h2_r2;
        self_data->grad_color += rel_pos * gradient_weight;
        self_data->lplc_color += weighted_h2
            * (distance_squared - 0.75f * h2_r2);
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
template <bool kSmsOnly>
__global__ __launch_bounds__(kSmsBlockThreads, kSmsMinBlocksPerSm)
void computeForceHybridKernel(
    int *micro_cell_offsets, ParticleIdxRange tra_range,
    ParticleBufferList buffers, int *compact_indices, int *cell_offsets,
    int *cell_particle_counts, const BlockTask *block_tasks,
    const int *device_sms_task_count, const int *device_middle)
{
    int tra_block_count = 0;
    if constexpr (!kSmsOnly)
    {
        // Device-side TRA/SMS split: the host over-provisions the grid and excess
        // blocks exit immediately, so the frame needs no host readback/synchronization.
        int tra_particle_count = __ldg(device_middle);
        if (tra_particle_count < 0 || tra_particle_count > tra_range.end)
            tra_particle_count = tra_range.end;
        tra_block_count = ceil_int(
            tra_particle_count - tra_range.begin, kSmsBlockThreads);

        if (blockIdx.x < tra_block_count)
        {
            int sorted_index = threadIdx.x
                + __umul24(blockIdx.x, blockDim.x) + tra_range.begin;
            if (sorted_index >= tra_particle_count) return;

            const int self_idx = __ldg(&compact_indices[sorted_index]);
            CFData self_data;
            self_data.pos = __ldg(&buffers.position_d[self_idx]);
            self_data.ev = __ldg(&buffers.evaluated_velocity[self_idx]);
            self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
            self_data.lplc_color = 0.0f;

            const ushort3 micro_cell = ParticlePos2CellPosM(
                self_data.pos, kDevSysPara.inv_cell_size);
            const ushort3 coarse_cell = calCI(micro_cell);
            const int local_micro_x = micro_cell.x & 3;
            float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
            float3 viscosity_sum = make_float3(0.0f, 0.0f, 0.0f);

            for (int neighbor_row = 0; neighbor_row < 9; ++neighbor_row)
            {
                const ParticleIdxRange neighbor_range =
                    findTraNeighborParticleRange(
                        micro_cell_offsets, cell_offsets,
                        cell_particle_counts, coarse_cell,
                        local_micro_x, neighbor_row % 3 - 1,
                        neighbor_row / 3 - 1, kDevSysPara.grid_size);
                knComputeCellForceTRA9(
                    &pressure_sum, &viscosity_sum, buffers, &self_data,
                    neighbor_range.begin,
                    neighbor_range.end - neighbor_range.begin);
            }

            float3 total_force = pressure_sum * kDevSysPara.spiky_value / 2
                + viscosity_sum * kDevSysPara.viscosity
                * kDevSysPara.visco_value;
            self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
            self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;
            self_data.lplc_color *= self_data.pos.w;

            const float surface_normal = sqrtf(
                self_data.grad_color.x * self_data.grad_color.x
                + self_data.grad_color.y * self_data.grad_color.y
                + self_data.grad_color.z * self_data.grad_color.z);
            float3 surface_force = make_float3(0.0f, 0.0f, 0.0f);
            if (surface_normal > kDevSysPara.surface_normal)
            {
                surface_force = self_data.grad_color
                    * kDevSysPara.surface_coe * self_data.lplc_color
                    / surface_normal;
            }

            total_force *= kDevSysPara.mass;
            buffers.acceleration[self_idx] = total_force + surface_force;
            return;
        }
    }
    {

        int sms_block_index = blockIdx.x - tra_block_count;
        if (sms_block_index * kSmsTasksPerBlock >= __ldg(device_sms_task_count))
            return;
        int task_index = sms_block_index * kSmsTasksPerBlock
            + (threadIdx.x >> 5);
        const BlockTask task = block_tasks[task_index];

        int cell_id = task.cellid;
        ushort3 cell_pos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);

        int cell_begin = __ldg(&cell_offsets[cell_id]);
        int cell_particle_count = __ldg(&cell_particle_counts[cell_id]);
        int self_idx = cell_begin + task.p_offset
            + (threadIdx.x & (kSmsTaskParticles - 1));

        int cell_end = cell_begin + cell_particle_count;
        const bool active = self_idx < cell_end;

        float3 pressure_sum = make_float3(0.0f, 0.0f, 0.0f);
        float3 viscosity_sum = make_float3(0.0f, 0.0f, 0.0f);
        CFData self_data;

#if GSPH_USE_REGISTER_SMS
        // Keep the warp-uniform inner loop branch-free while making inactive
        // tail lanes well-defined. Their accumulated result is discarded.
        const int safe_self_idx = active ? self_idx : cell_end - 1;
        self_data.pos = __ldg(&buffers.position_d[safe_self_idx]);
        self_data.ev = __ldg(&buffers.evaluated_velocity[safe_self_idx]);
        self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
        self_data.lplc_color = 0.0f;
#else
        if (active)
        {
            self_data.pos = __ldg(&buffers.position_d[self_idx]);
            self_data.ev = __ldg(&buffers.evaluated_velocity[self_idx]);
            self_data.grad_color = make_float3(0.0f, 0.0f, 0.0f);
            self_data.lplc_color = 0.0f;
        }
#endif

#if GSPH_USE_REGISTER_SMS
        // Each warp owns one independent task and exchanges neighbor data only
        // within its 32 lanes.
        {
            const bool warp_has_work = task.p_offset < cell_particle_count;
            __shared__ SmsRegisterTaskIterator neighbor_iterator;
            __shared__ float4 neighbor_positions[kSmsBlockThreads];
            __shared__ float4 neighbor_velocities[kSmsBlockThreads];
            int iterator_cell;
            int iterator_offset;
            int iterator_segment;
            neighbor_iterator.initialize(
                task.xxi, task.xxx, task.yyi, task.yyy, task.zzi, task.zzz,
                micro_cell_offsets, cell_pos, kDevSysPara.grid_size,
                iterator_cell, iterator_offset, iterator_segment);

            if (warp_has_work)
            {
                while (true)
                {
                    const int task_lane = threadIdx.x & (kSmsTaskParticles - 1);
                    int neighbor_begin = 0;
                    int neighbor_count = neighbor_iterator.nextBatch(
                        micro_cell_offsets, neighbor_begin,
                        iterator_cell, iterator_offset, iterator_segment);
                    if (neighbor_count == 0) break;

                    if (task_lane < neighbor_count)
                    {
                        const int neighbor_idx = neighbor_begin + task_lane;
                        neighbor_positions[threadIdx.x] = __ldg(
                            &buffers.position_d[neighbor_idx]);
                        neighbor_velocities[threadIdx.x] = __ldg(
                            &buffers.evaluated_velocity[neighbor_idx]);
                    }
                    const int task_base = threadIdx.x - task_lane;
                    __syncwarp(kFullWarpMask);
                    accumulateForceRegisterBatch(
                        neighbor_positions, neighbor_velocities, task_base,
                        &pressure_sum, &viscosity_sum, &self_data,
                        neighbor_count);
                    __syncwarp(kFullWarpMask);
                }
            }
        }
#else
        // Original shared-memory SMS path for both same-cell and different-cell tasks.
        constexpr int is_same = 0;
        __shared__ SimForSharedData128 shared_data;
        shared_data.initialize(task.zzi, task.zzz, task.xxi, task.xxx,
                               micro_cell_offsets, is_same, cell_offsets, cell_particle_counts,
                               cell_pos, kDevSysPara.grid_size);
        while (true)
        {
            __syncthreads();
            int neighbor_count = shared_data.read32Data(
                micro_cell_offsets, is_same, buffers);
            __syncthreads();
            if (neighbor_count == 0) break;
            if (active)
            {
                knComputeCellForceSMS64(
                    is_same, &pressure_sum, &viscosity_sum,
                    &shared_data, &self_data, neighbor_count);
            }
        }
#endif

        if (active)
        {
            float3 total_force = pressure_sum * kDevSysPara.spiky_value / 2
                + viscosity_sum * kDevSysPara.viscosity * kDevSysPara.visco_value;

            self_data.grad_color *= kDevSysPara.grad_poly6 * kDevSysPara.mass;
            self_data.lplc_color *= kDevSysPara.lplc_poly6 * kDevSysPara.mass;

            // position_d.w, loaded into self_data.pos.w, stores 1/density.
            self_data.lplc_color *= self_data.pos.w;
            float surface_normal_squared =
                self_data.grad_color.x * self_data.grad_color.x
                + self_data.grad_color.y * self_data.grad_color.y
                + self_data.grad_color.z * self_data.grad_color.z;
            float inverse_surface_normal = rsqrtf(surface_normal_squared);
            float surface_normal =
                surface_normal_squared * inverse_surface_normal;

            float3 surface_force;
            if (surface_normal > kDevSysPara.surface_normal)
            {
                surface_force = self_data.grad_color
                    * (kDevSysPara.surface_coe * self_data.lplc_color
                       * inverse_surface_normal);
            }
            else
            {
                surface_force = make_float3(0.0f, 0.0f, 0.0f);
            }

            total_force *= kDevSysPara.mass;
            buffers.acceleration[self_idx] = total_force + surface_force;
        }
    }
}

void launchForceHybridKernel(
    int block_count, bool sms_only, int *micro_cell_offsets,
    ParticleIdxRange tra_range, ParticleBufferList buffers,
    int *compact_indices, int *cell_offsets, int *cell_particle_counts,
    const BlockTask *block_tasks, const int *device_sms_task_count,
    const int *device_middle)
{
    if (sms_only)
    {
        computeForceHybridKernel<true><<<block_count, kSmsBlockThreads>>>(
            micro_cell_offsets, tra_range, buffers, compact_indices,
            cell_offsets, cell_particle_counts, block_tasks,
            device_sms_task_count, device_middle);
    }
    else
    {
        computeForceHybridKernel<false><<<block_count, kSmsBlockThreads>>>(
            micro_cell_offsets, tra_range, buffers, compact_indices,
            cell_offsets, cell_particle_counts, block_tasks,
            device_sms_task_count, device_middle);
    }
}
}

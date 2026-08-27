// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

// Switch for the density SMS different-cell path:
// 0 = original shared-memory path for both same-cell and different-cell tasks
// 1 = register-load + warp-local shared exchange for different-cell tasks,
//     shared-memory path kept for same-cell tasks (dominant case)
#ifndef DENSITY_SMS_USE_REGISTER_PATH
#define DENSITY_SMS_USE_REGISTER_PATH 1
#endif

namespace sph {
__device__
inline void knComputeCellDensitySMS64(const int& isSame, SimDenSharedData128 *sdata, CDAPData *self_data, int read_num)
{
    //   register float total_cell_density = 0;
    int kk = (1-isSame)*(threadIdx.x>>5);
    for (int i = (kk <<5); i < (kk<<5) + read_num; ++i)
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
    for (int i = 0; i < read_num; ++i)
    {
        float4 neighbor_position = sdata->getPos(i);
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        self_data->pos.w += powf_3(kDevSysPara.kernel_2 - dis_2);
    }

    //   return total_cell_density;
}
// WARNING: dead/broken kernel kept for reference only — cell_id and the shared-memory
// staging calls are commented out, so launching it is undefined behavior.
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
// WARNING: dead/broken kernel kept for reference only — cell_id and the shared-memory
// staging calls are commented out, so launching it is undefined behavior.
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
        data.pos = buff_list.position_d[self_idx];
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
__device__
void knComputeCellDensityTRA(ParticleBufferList &buff_list, CDAPData *self_data, int *cell_offset, int *cell_num, ushort3 cell_pos)
{
    //    float total_density = 0.0f;

    int cell_id = CellPos2CellIdx(cell_pos, kDevSysPara.grid_size);
    if (kInvalidCellIdx == cell_id) return;

    int start_idx = cell_offset[cell_id];
    int end_idx = start_idx + cell_num[cell_id];

    if (0xffffffff == start_idx) return;

    for (int i = start_idx; i < end_idx; ++i)
    {
        float4 neighbor_pos = __ldg(&buff_list.position_d[i]);
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
    for (int i = cell_offset; i < end_idx; ++i)
    {
        float4 neighbor_pos = __ldg(&buff_list.position_d[i]);

        float3 rel_pos = cal_rePos(neighbor_pos, self_data->pos);
        float dis_2 = rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y + rel_pos.z * rel_pos.z;

        if (dis_2 < kFloatSmall || dis_2 > kDevSysPara.kernel_2) continue;

        float h2_r2 = kDevSysPara.kernel_2 - dis_2;
        self_data->pos.w += h2_r2 * h2_r2 * h2_r2;
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

    for (int i = start_idx; i < end_idx; ++i)
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
    self_data.pos = buff_list.position_d[self_idx];
    //    self_data.pos = buff_list.position_d[self_idx];

    self_data.pos.w = 0;


    ushort3 cell_pos = ParticlePos2CellPos(self_data.pos, kDevSysPara.inv_cell_size);

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

#if DENSITY_SMS_USE_REGISTER_PATH
__device__ __forceinline__
void knComputeCellDensityReg64(float4 *shared_pos, int warp_base, CDAPData *self_data, int read_num)
{
    // Ensure the loop bound is uniform across the warp.
    read_num = __shfl_sync(0xFFFFFFFF, read_num, 0);
    #pragma unroll 4
    for (int src = 0; src < read_num; ++src)
    {
        float4 neighbor_position = shared_pos[warp_base + src];
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        float h2_r2 = kDevSysPara.kernel_2 - dis_2;
        self_data->pos.w += h2_r2 * h2_r2 * h2_r2;
    }
}
#endif

__global__ __launch_bounds__(64, 10)
void kncomputeDensityHybrid128n(int *cell_offset_M, ParticleIdxRange range, ParticleBufferList buff_list, int *cindex, int *cell_offset, int *cell_num, BlockTask *block_task, const int *d_num_block, const int *d_middle)
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

        register CDAPData self_data;
        self_data.pos = __ldg(&buff_list.position_d[self_idx]);
  //      self_data.pos =buff_list.position_d[self_idx];
        self_data.pos.w = 0;
        ushort3 cell_posc = ParticlePos2CellPosM(self_data.pos, kDevSysPara.inv_cell_size);

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
					kInvalidCellIdx == nid_left ? __ldg(&cell_offset[nid_mid]) : __ldg(&cell_offset_M[(nid_left << 6) + (xxx << 4)]);
				cell_nump_ = __ldg(&cell_num[nid_mid]);
				if (kInvalidCellIdx != nid_left) cell_nump_ += __ldg(&cell_offset[nid_mid]) - __ldg(&cell_offset_M[(nid_left << 6) + (xxx << 4)]);
				if (xxx == 3){
					if (kInvalidCellIdx != nid_right) cell_nump_ += __ldg(&cell_num[nid_right]);
				}
				else{
					if (kInvalidCellIdx != nid_right) cell_nump_ += __ldg(&cell_offset_M[(nid_right << 6) + ((xxx + 1) << 4)]) - __ldg(&cell_offset_M[(nid_right << 6)]);
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
        buff_list.evaluated_velocity[self_idx].w = (powf_7(__fdividef(self_data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;

		float denv = (5000 - self_data.pos.w) / 6000;
		buff_list.color[self_idx] = COLORA(1.0f*denv, 0.f, 1.0*denv, 1.0);
    }
    else{

        int t = blockIdx.x - bt_offset;
        if ((t << 1) >= __ldg(d_num_block)) return;  // over-provisioned SMS block (block-uniform exit)
		int n = (t << 1) + (threadIdx.x >> 5);
        BlockTask bt = block_task[n];
        int isSame = 0;// bt.isSame;

        int cell_id = bt.cellid;
		ushort3 cellpos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);

        register int cell_off = __ldg(&cell_offset[cell_id]);
        register int cell_np = __ldg(&cell_num[cell_id]);
        register int self_idx = cell_off + bt.p_offset + threadIdx.x % 32;

        register int temp_cell_end = cell_off + cell_np;

        register CDAPData data;

        bool active = (self_idx < temp_cell_end);
        if (active)   // initialize self data
        {
            data.pos = __ldg(&buff_list.position_d[self_idx]);
            data.pos.w = 0;
        }

#if DENSITY_SMS_USE_REGISTER_PATH
        // Register-load + warp-local shared exchange for different-cell tasks.
        // (The isSame==1 shared-memory variant was removed: isSame is hard-wired to 0.)
        {
            // A warp whose p_offset is past the cell has no self-particles.
            // All threads still participate in initialize() because it uses
            // __syncthreads(), but the empty warp can skip the iteration loop.
            bool warp_has_work = (bt.p_offset < cell_np);

            __shared__ SimDenRegData128 sdata;
			sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
            __shared__ float4 shared_pos[64];
            if (warp_has_work)
            {
                while (true)
                {
				    float4 my_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				    int r = sdata.read32DataReg(cell_offset_M, isSame, buff_list, my_pos);
                    if (0 == r) break;  // neighbor cells read complete
                    shared_pos[threadIdx.x] = my_pos;
                    __syncwarp();
                    int warp_base = (threadIdx.x >> 5) << 5;
				    knComputeCellDensityReg64(shared_pos, warp_base, &data, r);
                    __syncwarp();
                }
            }
        }
#else
        // Original shared-memory SMS path for both same-cell and different-cell tasks.
        __shared__ SimDenSharedData128 sdata;
		sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
        while (true)
        {
			__syncthreads();
			int r = sdata.read32Data(cell_offset_M, isSame, buff_list);
			__syncthreads();
            if (0 == r) break;  // neighbor cells read complete
            if (active)
            {
				knComputeCellDensitySMS64(isSame, &sdata, &data, r);
            }
        }
#endif

        if (self_idx < temp_cell_end)
        {
            data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
            data.pos.w += kDevSysPara.self_density;
            buff_list.position_d[self_idx].w = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
            buff_list.evaluated_velocity[self_idx].w = (powf_7(__fdividef(data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;

			float denv = (5000 - data.pos.w) / 6000;
			buff_list.color[self_idx] = COLORA(1.0f*denv, 1.0f*denv, 0.f, 1.0);
        }
    }
}

}

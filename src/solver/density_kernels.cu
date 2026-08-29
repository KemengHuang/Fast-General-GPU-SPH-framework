// Auto-generated from src/sph_kernel.cu during solver reorganization.

#include "solver/kernel_common.cuh"

// Select the live SMS implementation:
// 0 = legacy shared-memory iterator and neighbor staging
// 1 = register iterator state with warp-local neighbor staging
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
        float stored_density = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
        buff_list.position_d[self_idx].w = __fdividef(1.0f, stored_density);  // position_d.w stores 1/density
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
        float stored_density = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
        buff_list.position_d[self_idx].w = __fdividef(1.0f, stored_density);  // position_d.w stores 1/density
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
    buff_list.position_d[self_idx].w = __fdividef(1.0f, self_data.pos.w);  // position_d.w stores 1/density
    buff_list.evaluated_velocity[self_idx].w = (__powf(__fdividef(self_data.pos.w, kDevSysPara.rest_density), 7) - 1) * kDevSysPara.gas_constant;
}
__global__
void knComputeMixDensityTRA(ParticleBufferList buff_list, int *cell_offset, int *cell_num, ParticleIdxRange range)
{

}

#if GSPH_USE_REGISTER_SMS
__device__ __forceinline__
void accumulateDensityRegisterBatch(const float4 *neighbor_positions,
                                    int warp_base, CDAPData *self_data,
                                    int neighbor_count)
{
    #pragma unroll 4
    for (int src = 0; src < neighbor_count; ++src)
    {
        float4 neighbor_position = neighbor_positions[warp_base + src];
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        float h2_r2 = kDevSysPara.kernel_2 - dis_2;
        self_data->pos.w += h2_r2 * h2_r2 * h2_r2;
    }
}
#endif

template <bool kSmsOnly>
__global__ __launch_bounds__(kSmsBlockThreads, kSmsMinBlocksPerSm)
void computeDensityHybridKernel(
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
            CDAPData self_data;
            self_data.pos = __ldg(&buffers.position_d[self_idx]);
            self_data.pos.w = 0;

            const ushort3 micro_cell = ParticlePos2CellPosM(
                self_data.pos, kDevSysPara.inv_cell_size);
            const ushort3 coarse_cell = calCI(micro_cell);
            const int local_micro_x = micro_cell.x & 3;

            for (int neighbor_row = 0; neighbor_row < 9; ++neighbor_row)
            {
                const ParticleIdxRange neighbor_range =
                    findTraNeighborParticleRange(
                        micro_cell_offsets, cell_offsets,
                        cell_particle_counts, coarse_cell,
                        local_micro_x, neighbor_row % 3 - 1,
                        neighbor_row / 3 - 1, kDevSysPara.grid_size);
                knComputeCellDensityTRA9(
                    buffers, &self_data, neighbor_range.begin,
                    neighbor_range.end - neighbor_range.begin);
            }

            self_data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
            self_data.pos.w += kDevSysPara.self_density;
            buffers.position_d[self_idx].w =
                __fdividef(1.0f, self_data.pos.w);
            buffers.evaluated_velocity[self_idx].w =
                (powf_7(__fdividef(
                    self_data.pos.w, kDevSysPara.rest_density)) - 1)
                * kDevSysPara.gas_constant;

            const float density_color = (5000 - self_data.pos.w) / 6000;
            buffers.color[self_idx] =
                COLORA(density_color, 0.0f, density_color, 1.0f);
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

        CDAPData self_data;

#if GSPH_USE_REGISTER_SMS
        // Partial tasks still execute warp-uniform neighbor loops.  Give their
        // inactive lanes a valid duplicate self particle so the hot loop does
        // not need a per-batch activity branch and never reads uninitialized
        // register state.
        const int safe_self_idx = active ? self_idx : cell_end - 1;
        self_data.pos = __ldg(&buffers.position_d[safe_self_idx]);
        self_data.pos.w = 0;
#else
        if (active)
        {
            self_data.pos = __ldg(&buffers.position_d[self_idx]);
            self_data.pos.w = 0;
        }
#endif

#if GSPH_USE_REGISTER_SMS
        // Each warp owns one independent task and exchanges neighbor data only
        // within its 32 lanes.
        {
            // A warp whose p_offset is past the cell has no self-particles.
            // All threads still participate in initialize() because it uses
            // __syncthreads(), but the empty warp can skip the iteration loop.
            const bool warp_has_work = task.p_offset < cell_particle_count;

            __shared__ SmsRegisterTaskIterator neighbor_iterator;
            int iterator_cell;
            int iterator_offset;
            int iterator_segment;
            neighbor_iterator.initialize(
                task.xxi, task.xxx, task.yyi, task.yyy, task.zzi, task.zzz,
                micro_cell_offsets, cell_pos, kDevSysPara.grid_size,
                iterator_cell, iterator_offset, iterator_segment);
            __shared__ float4 neighbor_positions[kSmsBlockThreads];
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
                        neighbor_positions[threadIdx.x] = __ldg(
                            &buffers.position_d[neighbor_begin + task_lane]);
                    }
                    const int task_base = threadIdx.x - task_lane;
                    __syncwarp(kFullWarpMask);
                    accumulateDensityRegisterBatch(neighbor_positions, task_base, &self_data, neighbor_count);
                    __syncwarp(kFullWarpMask);
                }
            }
        }
#else
        // Original shared-memory SMS path for both same-cell and different-cell tasks.
        constexpr int is_same = 0;
        __shared__ SimDenSharedData128 shared_data;
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
                knComputeCellDensitySMS64(
                    is_same, &shared_data, &self_data, neighbor_count);
            }
        }
#endif

        if (active)
        {
            self_data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
            self_data.pos.w += kDevSysPara.self_density;
            float stored_density = self_data.pos.w < kFloatSmall ? kDevSysPara.rest_density : self_data.pos.w;
            buffers.position_d[self_idx].w = __fdividef(1.0f, stored_density);  // position_d.w stores 1/density
            buffers.evaluated_velocity[self_idx].w =
                (powf_7(__fdividef(
                    self_data.pos.w, kDevSysPara.rest_density)) - 1)
                * kDevSysPara.gas_constant;

            float density_color = (5000 - self_data.pos.w) / 6000;
            buffers.color[self_idx] = COLORA(
                density_color, density_color, 0.0f, 1.0f);
        }
    }
}

void launchDensityHybridKernel(
    int block_count, bool sms_only, int *micro_cell_offsets,
    ParticleIdxRange tra_range, ParticleBufferList buffers,
    int *compact_indices, int *cell_offsets, int *cell_particle_counts,
    const BlockTask *block_tasks, const int *device_sms_task_count,
    const int *device_middle)
{
    if (sms_only)
    {
        computeDensityHybridKernel<true><<<block_count, kSmsBlockThreads>>>(
            micro_cell_offsets, tra_range, buffers, compact_indices,
            cell_offsets, cell_particle_counts, block_tasks,
            device_sms_task_count, device_middle);
    }
    else
    {
        computeDensityHybridKernel<false><<<block_count, kSmsBlockThreads>>>(
            micro_cell_offsets, tra_range, buffers, compact_indices,
            cell_offsets, cell_particle_counts, block_tasks,
            device_sms_task_count, device_middle);
    }
}

}

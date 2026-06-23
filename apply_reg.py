import re
from pathlib import Path

root = Path('C:/Users/KemengHuang/Desktop/sph/Fast-General-GPU-SPH-framework')
common_path = root / 'src/solver/kernel_common.cuh'
density_path = root / 'src/solver/density_kernels.cu'

# ---------- 1. Add SimDenRegData128 to kernel_common.cuh ----------
common_text = common_path.read_text(encoding='utf-8')

# Locate SimDenSharedData128 block
start = common_text.find('class SimDenSharedData128')
assert start != -1
end = common_text.find('class SimForSharedData', start)
assert end != -1
block = common_text[start:end]

# Create register version
reg_block = block.replace('class SimDenSharedData128', 'class SimDenRegData128')
# Change read32Data signature and body
reg_block = reg_block.replace(
    '__device__ int read32Data(int *celloffM, const int& isSame, const ParticleBufferList& buff_list) {',
    '__device__ int read32DataReg(int *celloffM, const int& isSame, const ParticleBufferList& buff_list, float4& my_pos) {'
)
# Replace writes to shared position array with write to caller register
reg_block = reg_block.replace(
    'position_[idx] = buff_list.position_d[read_idx];',
    'my_pos = buff_list.position_d[read_idx];'
)
# Remove getPos accessor
reg_block = reg_block.replace(
    '    __device__  float4 & getPos(uint idx) {\n\t\treturn position_[idx];\n    }\n',
    ''
)
# Remove position_ member
reg_block = reg_block.replace(
    '    float4 position_[kNumSharedData * rate];\n',
    ''
)

new_common = common_text[:end] + '\n' + reg_block + common_text[end:]
common_path.write_text(new_common, encoding='utf-8')

# ---------- 2. Modify density_kernels.cu SMS path ----------
density_text = density_path.read_text(encoding='utf-8')

# Add shfl_float4 helper and register compute function before kncomputeDensityHybrid128n
helpers = r'''
__device__ __forceinline__
float4 shfl_float4(unsigned int mask, float4 v, int srcLane)
{
    v.x = __shfl_sync(mask, v.x, srcLane);
    v.y = __shfl_sync(mask, v.y, srcLane);
    v.z = __shfl_sync(mask, v.z, srcLane);
    v.w = __shfl_sync(mask, v.w, srcLane);
    return v;
}

__device__ __forceinline__
void knComputeCellDensityReg64(float4 my_pos, CDAPData *self_data, int read_num)
{
    for (int src = 0; src < read_num; ++src)
    {
        float4 neighbor_position = shfl_float4(0xFFFFFFFF, my_pos, src);
        float dis_2 = distance_square(self_data->pos, neighbor_position);
        if (kDevSysPara.kernel_2 < dis_2 || kFloatSmall > dis_2)
            continue;
        self_data->pos.w += powf_3(kDevSysPara.kernel_2 - dis_2);
    }
}

'''

# Insert helpers before __global__ //__launch_bounds__(64, 10) for kncomputeDensityHybrid128n
hybrid_marker = '__global__ //__launch_bounds__(64, 10)\nvoid kncomputeDensityHybrid128n'
assert hybrid_marker in density_text
idx = density_text.find(hybrid_marker)
density_text = density_text[:idx] + helpers + density_text[idx:]

# Replace SMS path block
old_sms = '''        int t = blockIdx.x - bt_offset;
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
            data.pos = __ldg(&buff_list.position_d[self_idx]);
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
		//	if (r > 32 && r < 64 && (threadIdx.x == 0 || threadIdx.x == 32)) printf("%d\\n", r);
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

			float denv = (5000 - data.pos.w) / 6000;
			if (isSame == 1){
				buff_list.color[self_idx] = COLORA(0.f, 1.0f*denv, 1.0*denv, 1.0);
			}
			else{
				buff_list.color[self_idx] = COLORA(1.0f*denv, 1.0f*denv, 0.f, 1.0);
			}


        }'''

new_sms = '''        int t = blockIdx.x - bt_offset;
		int n = (t << 1) + (threadIdx.x >> 5);
        BlockTask bt = block_task[n];
		// Register-shuffle path only handles one warp's own loaded set;
		// cross-warp sharing would still need shared memory for positions.
		int isSame = 0;

        int cell_id = bt.cellid;
		ushort3 cellpos = CellIdx2CellPos(cell_id, kDevSysPara.grid_size);

        register int self_idx = cell_offset[cell_id] + bt.p_offset + threadIdx.x % 32;

        register int temp_cell_end = cell_offset[cell_id] + cell_num[cell_id];

        register CDAPData data;

        if (self_idx < temp_cell_end)   // initialize self data
        {
            data.pos = __ldg(&buff_list.position_d[self_idx]);
            data.pos.w = 0;
        }
        __shared__ SimDenRegData128 sdata;
		sdata.initialize(bt.zzi, bt.zzz, bt.xxi, bt.xxx, cell_offset_M, isSame, cell_offset, cell_num, cellpos, kDevSysPara.grid_size);
        while (true)
        {
			__syncthreads();
			float4 my_pos = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			int r = sdata.read32DataReg(cell_offset_M, isSame, buff_list, my_pos);
			__syncthreads();
            if (0 == r) break;  // neighbor cells read complete
            if (self_idx < temp_cell_end)
            {
				knComputeCellDensityReg64(my_pos, &data, r);
            }
        }
        if (self_idx < temp_cell_end)
        {
            data.pos.w *= kDevSysPara.mass * kDevSysPara.poly6_value;
            data.pos.w += kDevSysPara.self_density;
            buff_list.position_d[self_idx].w = data.pos.w < kFloatSmall ? kDevSysPara.rest_density : data.pos.w;
            buff_list.evaluated_velocity[self_idx].w = (powf_7(__fdividef(data.pos.w, kDevSysPara.rest_density)) - 1) * kDevSysPara.gas_constant;

			float denv = (5000 - data.pos.w) / 6000;
			buff_list.color[self_idx] = COLORA(1.0f*denv, 1.0f*denv, 0.f, 1.0);
        }'''

assert old_sms in density_text, 'old SMS block not found'
density_text = density_text.replace(old_sms, new_sms)

density_path.write_text(density_text, encoding='utf-8')
print('done')

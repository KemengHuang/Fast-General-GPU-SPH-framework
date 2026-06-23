//
// sph_kernel_shared_data.cuh
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _SOLVER_KERNEL_COMMON_CUH_
#define _SOLVER_KERNEL_COMMON_CUH_


#define LOG_NUM_BANKS_MINE	 5
#define CONFLICT_FREE_OFFSET_MINE(index) ((index) >> LOG_NUM_BANKS)


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/cuda_math.cuh"
#include "particle/particle_buffer.h"
#include "core/sph_utils.cuh"
#include "core/sph_parameter.h"

namespace sph
{
// Thread/grid defaults (originally in sph_kernel.cu)
const int kDefaultNumThreadSMS = 32;
const int kDefulatMinBlocksSMS = 10;
const int kDefaultNumThreadSMS2 = 64;
const int kDefaultNumThreadTRA = 128;
const int kNumberNeighborCells = 27;

// Constant system parameter: single definition lives in device_context.cu
extern __constant__ SystemParameter kDevSysPara;

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

#define COLORA(r,g,b,a)    ( (uint((a)*255.0f)<<24) | (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) )

__device__ __host__
inline ushort3 calCI(const ushort3& in){
    ushort3 outd;
    outd.x = in.x >> 2;
    outd.y = in.y >> 2;
    outd.z = in.z >> 2;
    return outd;
}



const int kNumSharedData = 32;
const int kNumNeighborCells = 64;


const int rate = 2;


// Texture reads removed: use buff_list.position_d / buff_list.evaluated_velocity directly

struct GrediData
{
    float3  pos;
    float3  gradP1;
    float3  gradP2;
    float3  gradA1;
    float3  gradA2;
    float   sP1;
    float   sP2;
    float   sA1;
    float   sA2;
};

struct DivergData
{
    float3  pos;
    float3  mixVelo;
    float  diveVm1;
    float  diveVm2;
    float  diveAVm1;
    float  diveAVm2;
    float3   sVm1;
    float3   sVm2;
    float   sA1;
    float   sA2;
};

struct DivergTenData
{
    float3  pos;
    float3  mixVelo;
    float   mixVisc;
    float   mixPre;

    float3  diveTD;
    float3  diveTDM;
    float3  divePre;

    float3   sVm1;
    float3   sVm2;
    float   sA1;
    float   sA2;
};

struct CDAPData
{
    float4  pos;
};

struct CFData
{
    float4  pos;
    float4  ev;
//    float   pres;
    float3  grad_color;
    float   lplc_color;
};

//sf struct PCISPH
struct pciCDAPData
{
	float3  predicted_pos;
};

struct pciCFData
{
	float3  pos;
	float   correction_pres;
};

class SimDenSharedData
{
public:
    __device__ SimDenSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < 9) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, idx % 3 - 1, idx / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[idx] = 0;
                cell_nump_[idx] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[idx] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[idx] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
       //     register float3 *tp
         //       register float3 *tp = (float3*)(buff_list.position_d + read_idx);

                position_[idx] = buff_list.position_d [read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float4 & getPos(uint idx) {
        return position_[idx];
    }
private:
    //  __device__ void addOneCellPos(ushort3 &cell_pos);

    float4 position_[kNumSharedData];
    int cell_offset_[9];
    int cell_nump_[9];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
};
class SimDenSharedData128
{
public:
    __device__ SimDenSharedData128(){}

	__device__ void initialize(const int& minz, const int& maxz, const int& min, const int& max, int *celloffM, const int& isSame, int *cell_offset, int *cell_nump, ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = (idx>>5);
        int bj = idx % 32+(isSame)*(bi<<5);
        if (bj < 9) {
            int kk = bi * 9 + bj;
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
			else {
				int nid_left, nid_mid, nid_right;
				nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
				cell_offset_[kk] =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : celloffM[(nid_left << 6) + (min << 4)];
				int nc = 1;
				int my_cell_nump = cell_nump[nid_mid];
				if (kInvalidCellIdx != nid_left) {
					my_cell_nump += cell_offset[nid_mid] - celloffM[(nid_left << 6) + (min << 4)];
					nc++;
				}//cell_nump[nid_left];
				if (max == 3){
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += cell_nump[nid_right];
						nc++;
					}
				}
				else{
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += celloffM[(nid_right << 6) + ((max + 1) << 4)] - celloffM[(nid_right << 6)];//cell_nump[nid_right];
						nc++;
					}
				}
				int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
				int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
				int leng = 4 + leftl + rightl;

				if (my_cell_nump > leng * 22){
					if (((minz > 0) && ((kk % 9) < 3))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump>0){
							int lay = 0;
							while ((celloffM[begin_ + ((lay + 1) << 4)] - celloffM[begin_ + (minz << 2) + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else if (((maxz < 3) && ((kk % 9) > 5))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump > 0){
							int lay = 0;
							while ((celloffM[begin_ + (lay << 4) + ((maxz + 1) << 2)] - celloffM[begin_ + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else{
						cell_begin[kk] = -1;
					}
				}
				else{
					cell_begin[kk] = -1;
				}
				cell_nump_[kk] = my_cell_nump;
			}
        }
		if (bj == 31){
			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, 0, 0);
			int nid_left, nid_mid, nid_right;
			nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
			int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
			int rightl = (kInvalidCellIdx == nid_right ? 0 : 1+max);
			length[bi] = 4 + leftl + rightl;
			iminz[bi] = minz;
			imaxz[bi] = maxz;
		}
		depth[idx] = 0;
		int noCombie = (1 - isSame);
		current_cell_index_[idx] = 9 + bi * 9 * noCombie;
        offset_in_cell_[idx] = 0;
		__syncthreads();
		for (int i = bi * 9 * noCombie; i < bi * 9 * noCombie + 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

	__device__ int read32Data(int *celloffM, const int& isSame, const ParticleBufferList& buff_list) {
		unsigned int idx = threadIdx.x;
		int bi = (idx >> 5);
		int bj = idx % 32 + (isSame)*(bi << 5);
		int noCombine = (1 - isSame);
		int readDataSize = kNumSharedData*(1 + isSame);
		int curr_cell_index = current_cell_index_[idx];
		if (9 + bi * 9 * noCombine <= curr_cell_index) return 0;
		int cell_begin_ = cell_begin[curr_cell_index];
		if (cell_begin_ == -1){
			int offset_in_cell = offset_in_cell_[idx];
			int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
			int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
			if (num_read > bj) {
				int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
				position_[idx] = __ldg(&buff_list.position_d[read_idx]);
			}
			if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
			else {
				int next_cell_idx = curr_cell_index + 1;
				while (next_cell_idx < 9 + bi * 9 * noCombine) {
					if (0 != cell_nump_[next_cell_idx]) break;
					++next_cell_idx;
				}
				current_cell_index_[idx] = next_cell_idx;
				offset_in_cell_[idx] = 0;
			}
			return num_read;
		}
		else{
			int min = iminz[bi*(1 - isSame)];
			int max = imaxz[bi*(1 - isSame)];
			int lgth = length[bi*(1 - isSame)];
			if (((min > 0) && ((curr_cell_index % 9) < 3))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + ((lay + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;
				int celloff = celloffM[cell_begin_ + (min << 2) + (lay << 4)];
				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + ((lay + 1) << 4)] - celloff - offset_in_cell;
				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					position_[idx] = __ldg(&buff_list.position_d[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + ((depth[idx] + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
			else if (((max < 3) && ((curr_cell_index % 9) > 5))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;


				int celloff = celloffM[cell_begin_ + (lay << 4)];

				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloff - offset_in_cell;

				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					position_[idx] = __ldg(&buff_list.position_d[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + (depth[idx] << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
		}
	}
    __device__  float4 & getPos(uint idx) {
		return position_[idx];
    }
private:
    float4 position_[kNumSharedData * rate];
    int cell_offset_[9 * rate];
    int cell_nump_[9 * rate];
	int cell_begin[9 * rate];
	uint offset_in_cell_[kNumSharedData * rate];
    char current_cell_index_[kNumSharedData * rate];
	char depth[kNumSharedData * rate];
	char length[rate];
	char iminz[rate];
	char imaxz[rate];
};
class SimDenRegData128
{
public:
    __device__ SimDenRegData128(){}

	__device__ void initialize(const int& minz, const int& maxz, const int& min, const int& max, int *celloffM, const int& isSame, int *cell_offset, int *cell_nump, ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = (idx>>5);
        int bj = idx % 32+(isSame)*(bi<<5);
        if (bj < 9) {
            int kk = bi * 9 + bj;
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
			else {
				int nid_left, nid_mid, nid_right;
				nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
				cell_offset_[kk] =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : celloffM[(nid_left << 6) + (min << 4)];
				int nc = 1;
				int my_cell_nump = cell_nump[nid_mid];
				if (kInvalidCellIdx != nid_left) {
					my_cell_nump += cell_offset[nid_mid] - celloffM[(nid_left << 6) + (min << 4)];
					nc++;
				}//cell_nump[nid_left];
				if (max == 3){
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += cell_nump[nid_right];
						nc++;
					}
				}
				else{
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += celloffM[(nid_right << 6) + ((max + 1) << 4)] - celloffM[(nid_right << 6)];//cell_nump[nid_right];
						nc++;
					}
				}
				int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
				int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
				int leng = 4 + leftl + rightl;

				if (my_cell_nump > leng * 22){
					if (((minz > 0) && ((kk % 9) < 3))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump>0){
							int lay = 0;
							while ((celloffM[begin_ + ((lay + 1) << 4)] - celloffM[begin_ + (minz << 2) + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else if (((maxz < 3) && ((kk % 9) > 5))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump > 0){
							int lay = 0;
							while ((celloffM[begin_ + (lay << 4) + ((maxz + 1) << 2)] - celloffM[begin_ + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else{
						cell_begin[kk] = -1;
					}
				}
				else{
					cell_begin[kk] = -1;
				}
				cell_nump_[kk] = my_cell_nump;
			}
        }
		if (bj == 31){
			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, 0, 0);
			int nid_left, nid_mid, nid_right;
			nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
			int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
			int rightl = (kInvalidCellIdx == nid_right ? 0 : 1+max);
			length[bi] = 4 + leftl + rightl;
			iminz[bi] = minz;
			imaxz[bi] = maxz;
		}
		depth[idx] = 0;
		int noCombie = (1 - isSame);
		current_cell_index_[idx] = 9 + bi * 9 * noCombie;
        offset_in_cell_[idx] = 0;
		__syncthreads();
		for (int i = bi * 9 * noCombie; i < bi * 9 * noCombie + 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

	__device__ int read32DataReg(int *celloffM, const int& isSame, const ParticleBufferList& buff_list, float4& my_pos) {
		unsigned int idx = threadIdx.x;
		int bi = (idx >> 5);
		int bj = idx % 32 + (isSame)*(bi << 5);
		int noCombine = (1 - isSame);
		int readDataSize = kNumSharedData*(1 + isSame);
		int curr_cell_index = current_cell_index_[idx];
		if (9 + bi * 9 * noCombine <= curr_cell_index) return 0;
		int cell_begin_ = cell_begin[curr_cell_index];
		if (cell_begin_ == -1){
			int offset_in_cell = offset_in_cell_[idx];
			int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
			int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
			if (num_read > bj) {
				int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
				my_pos = __ldg(&buff_list.position_d[read_idx]);
			}
			if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
			else {
				int next_cell_idx = curr_cell_index + 1;
				while (next_cell_idx < 9 + bi * 9 * noCombine) {
					if (0 != cell_nump_[next_cell_idx]) break;
					++next_cell_idx;
				}
				current_cell_index_[idx] = next_cell_idx;
				offset_in_cell_[idx] = 0;
			}
			return num_read;
		}
		else{
			int min = iminz[bi*(1 - isSame)];
			int max = imaxz[bi*(1 - isSame)];
			int lgth = length[bi*(1 - isSame)];
			if (((min > 0) && ((curr_cell_index % 9) < 3))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + ((lay + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;
				int celloff = celloffM[cell_begin_ + (min << 2) + (lay << 4)];
				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + ((lay + 1) << 4)] - celloff - offset_in_cell;
				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					my_pos = __ldg(&buff_list.position_d[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + ((depth[idx] + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
			else if (((max < 3) && ((curr_cell_index % 9) > 5))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;


				int celloff = celloffM[cell_begin_ + (lay << 4)];

				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloff - offset_in_cell;

				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					my_pos = __ldg(&buff_list.position_d[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + (depth[idx] << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
		}
	}
private:
    int cell_offset_[9 * rate];
    int cell_nump_[9 * rate];
	int cell_begin[9 * rate];
	uint offset_in_cell_[kNumSharedData * rate];
    char current_cell_index_[kNumSharedData * rate];
	char depth[kNumSharedData * rate];
	char length[rate];
	char iminz[rate];
	char imaxz[rate];
};
class SimForSharedData
{
public:
    __device__ SimForSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < 9) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, idx % 3 - 1, idx / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[idx] = 0;
                cell_nump_[idx] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[idx] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[idx] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
      //      register float3* tp = (float3*)(buff_list.position_d+read_idx);
            position_d[idx] = __ldg(&buff_list.position_d[read_idx]);
            ev_[idx] = __ldg(&buff_list.evaluated_velocity[read_idx]);
     //       pressure_[idx] = buff_list.pressure[read_idx];
       //     density_[idx] = buff_list.density[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__ float4 & getPosition(unsigned int idx) { return position_d[idx]; }

    __device__ float4 & getEV(unsigned int idx) { return ev_[idx]; }

    __device__ float & getPressure(unsigned int idx) { return pressure_[idx]; }

//    __device__ float & getDensity(unsigned int idx) { return density_[idx]; }
private:
    //  __device__ void addOneCellPos(ushort3 &cell_pos);

    float4 position_d[kNumSharedData];
    float4  ev_[kNumSharedData];
    float   pressure_[kNumSharedData];
//    float   density_[kNumSharedData];

    int cell_offset_[9];
    int cell_nump_[9];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
};

class SimForSharedData128
{
public:
    __device__ SimForSharedData128(){}

	__device__ void initialize(const int& minz, const int& maxz, const int& min, const int& max, int *celloffM, const int& isSame, int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
		int bi = (idx>>5);
		int bj = idx % 32 + (isSame)*(bi<<5);

        if (bj < 9) {

            int kk = bi * 9 + bj;

			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
            else {
				int nid_left, nid_mid, nid_right;
				nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
				cell_offset_[kk] =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : celloffM[(nid_left << 6) + (min << 4)];
				int nc = 1;
				int my_cell_nump = cell_nump[nid_mid];
				if (kInvalidCellIdx != nid_left) {
					my_cell_nump += cell_offset[nid_mid] - celloffM[(nid_left << 6) + (min << 4)];
					nc++;
				}//cell_nump[nid_left];
				if (max == 3){
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += cell_nump[nid_right];
						nc++;
					}
				}
				else{
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += celloffM[(nid_right << 6) + ((max + 1) << 4)] - celloffM[(nid_right << 6)];//cell_nump[nid_right];
						nc++;
					}
				}
				int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
				int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
				int leng = 4 + leftl + rightl;

				if (my_cell_nump > leng * 22){
					if (((minz > 0) && ((kk % 9) < 3))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump>0){
							int lay = 0;
							while ((celloffM[begin_ + ((lay + 1) << 4)] - celloffM[begin_ + (minz << 2) + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else if (((maxz < 3) && ((kk % 9) > 5))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump > 0){
							int lay = 0;
							while ((celloffM[begin_ + (lay << 4) + ((maxz + 1) << 2)] - celloffM[begin_ + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else{
						cell_begin[kk] = -1;
					}
				}
				else{
					cell_begin[kk] = -1;
				}
				cell_nump_[kk] = my_cell_nump;
			}
        }

		if (bj == 31){
			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, 0, 0);
			int nid_left, nid_mid, nid_right;
			nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
			int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
			int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
			length[bi] = 4 + leftl + rightl;
			iminz[bi] = minz;
			imaxz[bi] = maxz;
		}
		depth[idx] = 0;

		int noCombine = 1 - isSame;

		current_cell_index_[idx] = 9 + bi * 9 * noCombine;
        offset_in_cell_[idx] = 0;
		
		__syncthreads();
		for (int i = bi * 9 * noCombine; i < bi * 9 * noCombine + 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

	__device__ int read32Data(int *celloffM, const int& isSame, const ParticleBufferList& buff_list) {

		
        unsigned int idx = threadIdx.x;

		int bi = (idx >> 5);
		int bj = idx % 32 + (isSame)*(bi << 5);

        int curr_cell_index = current_cell_index_[idx];
		int noCombine = 1 - isSame;
		int readDataSize = kNumSharedData*(1 + isSame);
        // neighbors read complete
		if (9 + bi * 9 * noCombine <= curr_cell_index) return 0;
		int cell_begin_ = cell_begin[curr_cell_index];
		if (cell_begin_ == -1){
			int offset_in_cell = offset_in_cell_[idx];
			int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
			int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
			if (num_read > bj) {
				int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
				position_d[idx] = __ldg(&buff_list.position_d[read_idx]);
				ev_[idx] = __ldg(&buff_list.evaluated_velocity[read_idx]);
				//         position_d[idx] = buff_list.position_d[read_idx];
				//         ev_[idx] = buff_list.evaluated_velocity[read_idx];
			}
			if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
			else {
				int next_cell_idx = curr_cell_index + 1;
				while (next_cell_idx < 9 + bi * 9 * noCombine) {
					if (0 != cell_nump_[next_cell_idx]) break;
					++next_cell_idx;
				}
				current_cell_index_[idx] = next_cell_idx;
				offset_in_cell_[idx] = 0;
			}
			return num_read;
		}
		else{
			int min = iminz[bi*(1 - isSame)];
			int max = imaxz[bi*(1 - isSame)];
			int lgth = length[bi*(1 - isSame)];
			if (((min > 0) && ((curr_cell_index % 9) < 3))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + ((lay + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;
				int celloff = celloffM[cell_begin_ + (min << 2) + (lay << 4)];
				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + ((lay + 1) << 4)] - celloff - offset_in_cell;
				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					position_d[idx] = __ldg(&buff_list.position_d[read_idx]);
					ev_[idx] = __ldg(&buff_list.evaluated_velocity[read_idx]);
					//         position_d[idx] = buff_list.position_d[read_idx];
					//         ev_[idx] = buff_list.evaluated_velocity[read_idx];
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + ((depth[idx] + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
			else if (((max < 3) && ((curr_cell_index % 9) > 5))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;


				int celloff = celloffM[cell_begin_ + (lay << 4)];

				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloff - offset_in_cell;

				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					position_d[idx] = __ldg(&buff_list.position_d[read_idx]);
					ev_[idx] = __ldg(&buff_list.evaluated_velocity[read_idx]);
					//         position_d[idx] = buff_list.position_d[read_idx];
					//         ev_[idx] = buff_list.evaluated_velocity[read_idx];
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + (depth[idx] << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
		}
    }

	__device__ float4 & getPosition(unsigned int idx) { return position_d[idx]; }

	__device__ float4 & getEV(unsigned int idx) { return ev_[idx]; }

//    __device__ float & getPressure(unsigned int idx) { return pressure_[idx]; }

//    __device__ float & getDensity(unsigned int idx) { return density_[idx]; }

private:
    float4 position_d[kNumSharedData * rate];
	float4  ev_[kNumSharedData * rate];
    int cell_offset_[9 * rate];
    int cell_nump_[9 * rate];
	int cell_begin[9 * rate];
	uint offset_in_cell_[kNumSharedData * rate];
	char current_cell_index_[kNumSharedData * rate];
	char depth[kNumSharedData * rate];
	char length[rate];
	char iminz[rate];
	char imaxz[rate];
};

// Register-only metadata helper for the force SMS path.
// Neighbor positions and evaluated velocities are loaded into caller-owned
// registers; exchange is done through a small warp-level shared buffer.
class SimForRegData128
{
public:
    __device__ SimForRegData128(){}

	__device__ void initialize(const int& minz, const int& maxz, const int& min, const int& max, int *celloffM, const int& isSame, int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
		int bi = (idx>>5);
		int bj = idx % 32+(isSame)*(bi<<5);

        if (bj < 9) {

            int kk = bi * 9 + bj;

			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
            else {
				int nid_left, nid_mid, nid_right;
				nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
				++neighbor_pos.x;
				nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
				cell_offset_[kk] =
					kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : celloffM[(nid_left << 6) + (min << 4)];
				int nc = 1;
				int my_cell_nump = cell_nump[nid_mid];
				if (kInvalidCellIdx != nid_left) {
					my_cell_nump += cell_offset[nid_mid] - celloffM[(nid_left << 6) + (min << 4)];
					nc++;
				}
				if (max == 3){
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += cell_nump[nid_right];
						nc++;
					}
				}
				else{
					if (kInvalidCellIdx != nid_right) {
						my_cell_nump += celloffM[(nid_right << 6) + ((max + 1) << 4)] - celloffM[(nid_right << 6)];
						nc++;
					}
				}
				int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
				int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
				int leng = 4 + leftl + rightl;

				if (my_cell_nump > leng * 22){
					if (((minz > 0) && ((kk % 9) < 3))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump>0){
							int lay = 0;
							while ((celloffM[begin_ + ((lay + 1) << 4)] - celloffM[begin_ + (minz << 2) + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else if (((maxz < 3) && ((kk % 9) > 5))){
						int begin_ = kInvalidCellIdx == nid_left ? (nid_mid << 6) : (nid_left << 6) + (min << 4);
						cell_begin[kk] = begin_;
						if (my_cell_nump > 0){
							int lay = 0;
							while ((celloffM[begin_ + (lay << 4) + ((maxz + 1) << 2)] - celloffM[begin_ + (lay << 4)]) == 0){
								lay++;
								if (lay >= leng){
									my_cell_nump = 0;
								}
							}
						}
					}
					else{
						cell_begin[kk] = -1;
					}
				}
				else{
					cell_begin[kk] = -1;
				}
				cell_nump_[kk] = my_cell_nump;
			}
        }

		if (bj == 31){
			ushort3 neighbor_pos = cell_pos + make_ushort3(-1, 0, 0);
			int nid_left, nid_mid, nid_right;
			nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
			++neighbor_pos.x;
			nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
			int leftl = (kInvalidCellIdx == nid_left ? 0 : 4 - min);
			int rightl = (kInvalidCellIdx == nid_right ? 0 : 1 + max);
			length[bi] = 4 + leftl + rightl;
			iminz[bi] = minz;
			imaxz[bi] = maxz;
		}
		depth[idx] = 0;

		int noCombine = 1 - isSame;

		current_cell_index_[idx] = 9 + bi * 9 * noCombine;
        offset_in_cell_[idx] = 0;
		__syncthreads();
		for (int i = bi * 9 * noCombine; i < bi * 9 * noCombine + 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

	__device__ int read32DataReg(int *celloffM, const int& isSame, const ParticleBufferList& buff_list, float4& my_pos, float4& my_ev) {
        unsigned int idx = threadIdx.x;
		int bi = (idx >> 5);
		int bj = idx % 32 + (isSame)*(bi << 5);

        int curr_cell_index = current_cell_index_[idx];
		int noCombine = 1 - isSame;
		int readDataSize = kNumSharedData*(1 + isSame);
        // neighbors read complete
		if (9 + bi * 9 * noCombine <= curr_cell_index) return 0;
		int cell_begin_ = cell_begin[curr_cell_index];
		if (cell_begin_ == -1){
			int offset_in_cell = offset_in_cell_[idx];
			int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
			int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
			if (num_read > bj) {
				int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
				my_pos = __ldg(&buff_list.position_d[read_idx]);
				my_ev  = __ldg(&buff_list.evaluated_velocity[read_idx]);
			}
			if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
			else {
				int next_cell_idx = curr_cell_index + 1;
				while (next_cell_idx < 9 + bi * 9 * noCombine) {
					if (0 != cell_nump_[next_cell_idx]) break;
					++next_cell_idx;
				}
				current_cell_index_[idx] = next_cell_idx;
				offset_in_cell_[idx] = 0;
			}
			return num_read;
		}
		else{
			int min = iminz[bi*(1 - isSame)];
			int max = imaxz[bi*(1 - isSame)];
			int lgth = length[bi*(1 - isSame)];
			if (((min > 0) && ((curr_cell_index % 9) < 3))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + ((lay + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;
				int celloff = celloffM[cell_begin_ + (min << 2) + (lay << 4)];
				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + ((lay + 1) << 4)] - celloff - offset_in_cell;
				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					my_pos = __ldg(&buff_list.position_d[read_idx]);
					my_ev  = __ldg(&buff_list.evaluated_velocity[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + ((depth[idx] + 1) << 4)] - celloffM[cell_begin_ + (min << 2) + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
			else if (((max < 3) && ((curr_cell_index % 9) > 5))){
				int lay = depth[idx];
				while ((celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (lay << 4)]) == 0){
					lay++;
				}
				depth[idx] = lay;


				int celloff = celloffM[cell_begin_ + (lay << 4)];

				int offset_in_cell = offset_in_cell_[idx];
				int remain_nump = celloffM[cell_begin_ + (lay << 4) + ((max + 1) << 2)] - celloff - offset_in_cell;

				int num_read = remain_nump > readDataSize ? readDataSize : remain_nump;
				if (num_read > bj) {
					int read_idx = celloff + offset_in_cell + bj;
					my_pos = __ldg(&buff_list.position_d[read_idx]);
					my_ev  = __ldg(&buff_list.evaluated_velocity[read_idx]);
				}
				if (remain_nump > readDataSize) offset_in_cell_[idx] += readDataSize;
				else {
					depth[idx]++;
					if (depth[idx] < lgth){
						while ((celloffM[cell_begin_ + (depth[idx] << 4) + ((max + 1) << 2)] - celloffM[cell_begin_ + (depth[idx] << 4)]) == 0){
							depth[idx]++;
							if (depth[idx] >= lgth) break;
						}
					}
					if (depth[idx] >= lgth){
						int next_cell_idx = curr_cell_index + 1;
						while (next_cell_idx < 9 + bi * 9 * noCombine) {
							if (0 != cell_nump_[next_cell_idx]) break;
							++next_cell_idx;
						}
						current_cell_index_[idx] = next_cell_idx;
						depth[idx] = 0;
					}
					offset_in_cell_[idx] = 0;
				}
				return num_read;
			}
		}
    }

private:
    int cell_offset_[9 * rate];
    int cell_nump_[9 * rate];
	int cell_begin[9 * rate];
	uint offset_in_cell_[kNumSharedData * rate];
    char current_cell_index_[kNumSharedData * rate];
	char depth[kNumSharedData * rate];
	char length[rate];
	char iminz[rate];
	char imaxz[rate];
};











//sf struct PCISPH over-------------------------
class CfkSharedData128
{
public:
    __device__ CfkSharedData128(){}


    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < kNumNeighborCells)
        {

            int kk = bi * 27 + bj;

            cell_offset_[kk] = 0;
            cell_nump_[kk] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(bj % 3 - 1, bj / 3 % 3 - 1, bj / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[kk] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[kk] = cell_offset[neighbor_id];
                cell_nump_[kk] = cell_nump[neighbor_id];
            }
        }

        current_cell_index_[idx] = kNumNeighborCells;
        offset_in_cell_[idx] = 0;

        for (int i = bi * 27; i < bi * 27 + kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(ParticleBufferList &buff_list) {

        unsigned int idx = threadIdx.x;
        int bj = idx % 32;
        int bi = idx / 32;

        int curr_cell_index = current_cell_index_[idx];

        if (bi * 27 + kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
//            position[idx] = buff_list.position[read_idx];
            c_pressure[idx] = buff_list.correction_pressure[read_idx];
            pha_[idx] = buff_list.phase[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells + bi * 27) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
    }
    __device__  float3 & getPos(uint idx) {
        return position[idx];
    }
    __device__  float & getCor_press(uint idx) {
        return c_pressure[idx];
    }
    __device__  condition & get_pha(uint idx) {
        return pha_[idx];
    }

private:
    //__device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position[kNumSharedData * 2];
    float c_pressure[kNumSharedData * 2];
    condition pha_[kNumSharedData * 2];

    int cell_offset_[kNumNeighborCells * 2];
    int cell_nump_[kNumNeighborCells * 2];
    int offset_in_cell_[kNumSharedData * 2];
    int current_cell_index_[kNumSharedData * 2];
};


class CfkSharedData
{
public:
    __device__ CfkSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {//knInitializeSharedMemory2(/*out*/int *sm_cell_offset, int *sm_cell_nump, int &offset_in_cell, int &curr_cell_index,
        //  /*in*/int* cell_offset, int* cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells)
        {
            cell_offset_[idx] = 0;
            cell_nump_[idx] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[idx] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[idx] = cell_offset[neighbor_id];
                cell_nump_[idx] = cell_nump[neighbor_id];
            }
        }
        //        syncthreads1D();

        current_cell_index_[idx] = kNumNeighborCells;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(ParticleBufferList &buff_list) {
        /*unsigned int idx = threadIdx.x;
        int num_read;
        __syncthreads();
        if (kNumNeighborCells <= current_cell_index_) return 0;
        int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
        num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

        if (num_read > idx) {
        int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;
        pred_positions[idx] = pos[read_idx];
        phase_[idx] = pha[read_idx];
        }

        __syncthreads();


        if (0 == idx) {
        if (num_in_cell > num_read + offset_in_cell_) {
        offset_in_cell_ += num_read;
        }
        else {
        int i = current_cell_index_ + 1;
        for (; i < kNumNeighborCells; ++i) {
        if (kInvalidCellIdx != cell_start_index_[i]) break;
        }
        current_cell_index_ = i;
        offset_in_cell_ = 0;
        }
        }
        return num_read;*/
        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
//            position[idx] = buff_list.position[read_idx];
            c_pressure[idx] = buff_list.correction_pressure[read_idx];
            pha_[idx] = buff_list.phase[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
    }

    __device__  float3 & getPos(uint idx) {
        return position[idx];
    }
    __device__  float & getCor_press(uint idx) {
        return c_pressure[idx];
    }
    __device__  condition & get_pha(uint idx) {
        return pha_[idx];
    }

private:
    //__device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position[kNumSharedData];
    float c_pressure[kNumSharedData];
    condition pha_[kNumSharedData];

    int cell_offset_[kNumNeighborCells];
    int cell_nump_[kNumNeighborCells];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
};







class CKSharedData
{
public:
    __device__ CKSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < 9) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, idx % 3 - 1, idx / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[idx] = 0;
                cell_nump_[idx] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[idx] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[idx] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_cf(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
//            position_[idx] = buff_list.position[read_idx];
            c_pressure[idx] = buff_list.correction_pressure[read_idx];
            pha_[idx] = buff_list.phase[read_idx];
            //    ev[idx] = buff_list.evaluated_velocity[read_idx];
            //pressure[idx] = buff_list.pressure[read_idx];
            //density[idx] = buff_list.density[read_idx];//1.0f / buff_list.density[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPos(uint idx) {
        return position_[idx];
    }
    __device__  float & getCor_press(uint idx) {
        return c_pressure[idx];
    }
    __device__  condition & get_pha(uint idx) {
        return pha_[idx];
    }

private:
    //__device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData];
    float c_pressure[kNumSharedData];
    condition pha_[kNumSharedData];
    int cell_offset_[9];
    int cell_nump_[9];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
};









class CMDSharedData
{
public:
    __device__ CMDSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < 9) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, idx % 3 - 1, idx / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[idx] = 0;
                cell_nump_[idx] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[idx] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[idx] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_den(float3* pre_pos, condition *pha) {

        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
            position_[idx] = pre_pos[read_idx];
            pha_[idx] = pha[read_idx];
            //    ev[idx] = buff_list.evaluated_velocity[read_idx];
            //pressure[idx] = buff_list.pressure[read_idx];
            //density[idx] = buff_list.density[read_idx];//1.0f / buff_list.density[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPre_Pos(uint idx) {
        return position_[idx];
    }

private:
    //   __device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData];
    int cell_offset_[9];
    int cell_nump_[9];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
    condition pha_[kNumSharedData];
};



class CMFSharedData
{
public:
    __device__ CMFSharedData(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < 9) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, idx % 3 - 1, idx / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[idx] = 0;
                cell_nump_[idx] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[idx] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[idx] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_other_Force(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
 //           register float3 *tp = (float3*)(buff_list.position_d+read_idx);
 //           position_[idx] = *tp;
//            ev[idx] = buff_list.evaluated_velocity[read_idx];
            //pressure[idx] = buff_list.pressure[read_idx];
            //density[idx] = buff_list.density[read_idx];//1.0f / buff_list.density[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPos(uint idx) {
        return position_[idx];
    }
    __device__  float3 & getEV(uint idx) {
        return ev[idx];
    }

private:
    //  __device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData];
    float3 ev[kNumSharedData];


    int cell_offset_[9];
    int cell_nump_[9];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];
};






class CdapSharedData
{
public:
    __device__ CdapSharedData(){}

    __device__ void initialize(int *cell_of, int *cell_nm, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells) {
            ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id) {
                cell_start_index_[idx] = kInvalidCellIdx;
            }
            else {
                cell_start_index_[idx] = cell_of[neighbor_id];
                cell_end_index_[idx] = cell_of[neighbor_id] + cell_nm[neighbor_id];
            }
        }

        __syncthreads();

        if (0 == threadIdx.x) {
            current_cell_index_ = kNumNeighborCells;
            offset_in_cell_ = 0;

            for (int i = 0; i < kNumNeighborCells; ++i) {
                if (kInvalidCellIdx != cell_start_index_[i]) {
                    current_cell_index_ = i;
                    break;
                }
            }
        }
    }

    __device__ int read32Data(float3 *pos,condition *pha) {
        unsigned int idx = threadIdx.x;
        int num_read;

        __syncthreads();

        // neighbors read complete
        if (kNumNeighborCells <= current_cell_index_) return 0;

        // estimate nump will read
        int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
        num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

        if (num_read > idx) {
            int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;
            positions[idx] = pos[read_idx];
			phase_[idx] = pha[read_idx];
        }

        __syncthreads();

        // update info
        if (0 == idx) {
            // this cell has not been completed
            if (num_in_cell > num_read + offset_in_cell_) {
                offset_in_cell_ += num_read;
            }
            else { // new cell
                int i = current_cell_index_ + 1;
                for (; i < kNumNeighborCells; ++i) {
                    if (kInvalidCellIdx != cell_start_index_[i]) break;
                }
                current_cell_index_ = i;
                offset_in_cell_ = 0;
            }
        }
        return num_read;
    }

    __device__ float3 & getValue(unsigned int idx) {
        return positions[idx];
    }

	__device__ condition & getPhase(unsigned int idx) {
		return phase_[idx];
	}

private:
    __device__ void addOneCellPos(ushort3 &cell_pos);

    float3 positions[kNumSharedData];
	condition phase_[kNumSharedData];
    int cell_start_index_[kNumNeighborCells];
    int cell_end_index_[kNumNeighborCells];

    int offset_in_cell_;
    int current_cell_index_;
};

class CfSharedData
{
public:
    __device__ CfSharedData() {}

    __device__ void initialize(int *cell_of, int *cell_nb, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells)
        {
            ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_start_index_[idx] = kInvalidCellIdx;
            }
            else
            {
                cell_start_index_[idx] = cell_of[neighbor_id];
                cell_end_index_[idx] = cell_of[neighbor_id] + cell_nb[neighbor_id];
            }
        }

        __syncthreads();

        if (0 == threadIdx.x)
        {
            current_cell_index_ = kNumNeighborCells;
            offset_in_cell_ = 0;

            for (int i = 0; i < kNumNeighborCells; ++i)
            {
                if (kInvalidCellIdx != cell_start_index_[i])
                {
                    current_cell_index_ = i;
                    break;
                }
            }
        }
    }

    __device__ unsigned int read32Data(ParticleBufferList &buff_list) {
        unsigned int idx = threadIdx.x;
        int num_read;

        __syncthreads();

        // neighbors read complete
        if (kNumNeighborCells <= current_cell_index_) return 0;

        // estimate num will read
        int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
        num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

        if (num_read > idx)
        {
            // get index
            int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;

            // read data
//            position_[idx] = buff_list.position[read_idx];
//            ev_[idx] = buff_list.evaluated_velocity[read_idx];
            pressure_[idx] = buff_list.pressure[read_idx];
//            density_[idx] = buff_list.density[read_idx];
			phase_[idx] = buff_list.phase[read_idx];
        }

        __syncthreads();

        // update info
        if (0 == idx)
        {
            // this cell has not been completed
            if (num_in_cell > num_read + offset_in_cell_)
            {
                offset_in_cell_ += num_read;
            }
            else // new cell
            {
                int i = current_cell_index_ + 1;
                for (; i < kNumNeighborCells; ++i)
                {
                    if (kInvalidCellIdx != cell_start_index_[i])
                    {
                        break;
                    }
                }
                current_cell_index_ = i;
                offset_in_cell_ = 0;
            }
        }

        return num_read;
    }

    __device__ float3 & getPos(unsigned int idx) { return position_[idx]; }

    __device__ float3 & getEV(unsigned int idx) { return ev_[idx]; }

    __device__ float & getPressure(unsigned int idx) { return pressure_[idx]; }

    __device__ float & getDensity(unsigned int idx) { return density_[idx]; }

	__device__ condition & getPhase(unsigned int idx) { return phase_[idx]; }  //sf add

private:
    float3  position_[kNumSharedData];
    float3  ev_[kNumSharedData];
    float   pressure_[kNumSharedData];
    float   density_[kNumSharedData];
	condition phase_[kNumSharedData]; //sf add

    int cell_start_index_[kNumNeighborCells];
    int cell_end_index_[kNumNeighborCells];

    int offset_in_cell_;
    int current_cell_index_;
};

//sf class PCISPH------------------------------------------------------------------------


class pmfCdapSharedData
{
public:
    __device__ pmfCdapSharedData(){}

    /*__device__ void initialize(int *cell_of, int *cell_nm, const ushort3 &cell_pos, const ushort3 &grid_size) {
    unsigned int idx = threadIdx.x;

    if (idx < kNumNeighborCells) {
    cell_end_index_[idx] = 0;
    ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
    int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
    if (kInvalidCellIdx == neighbor_id) {
    cell_start_index_[idx] = kInvalidCellIdx;
    }
    else {
    cell_start_index_[idx] = cell_of[neighbor_id];
    cell_end_index_[idx] = cell_of[neighbor_id] + cell_nm[neighbor_id];
    }
    }

    __syncthreads();

    if (0 == threadIdx.x) {
    current_cell_index_ = kNumNeighborCells;
    offset_in_cell_ = 0;

    for (int i = 0; i < kNumNeighborCells; ++i) {
    if (0 != cell_end_index_[i]) {
    current_cell_index_ = i;
    break;
    }
    }
    }
    }*/
    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {//knInitializeSharedMemory2(/*out*/int *sm_cell_offset, int *sm_cell_nump, int &offset_in_cell, int &curr_cell_index,
        //  /*in*/int* cell_offset, int* cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells)
        {
            cell_offset_[idx] = 0;
            cell_nump_[idx] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[idx] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[idx] = cell_offset[neighbor_id];
                cell_nump_[idx] = cell_nump[neighbor_id];
            }
        }
        //        syncthreads1D();

        current_cell_index_[idx] = kNumNeighborCells;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(ParticleBufferList &buff_list) {
        /*unsigned int idx = threadIdx.x;
        int num_read;
        __syncthreads();
        if (kNumNeighborCells <= current_cell_index_) return 0;
        int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
        num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

        if (num_read > idx) {
        int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;
        pred_positions[idx] = pos[read_idx];
        phase_[idx] = pha[read_idx];
        }

        __syncthreads();


        if (0 == idx) {
        if (num_in_cell > num_read + offset_in_cell_) {
        offset_in_cell_ += num_read;
        }
        else {
        int i = current_cell_index_ + 1;
        for (; i < kNumNeighborCells; ++i) {
        if (kInvalidCellIdx != cell_start_index_[i]) break;
        }
        current_cell_index_ = i;
        offset_in_cell_ = 0;
        }
        }
        return num_read;*/
        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
//            positions[idx] = buff_list.position[read_idx];
//            ev[idx] = buff_list.evaluated_velocity[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
    }

    __device__ float3 & getPos(unsigned int idx) {
        return positions[idx];
    }

    __device__ float3 & getEV(unsigned int idx) {
        return ev[idx];
    }

private:
    float3 positions[kNumSharedData];
    float3 ev[kNumSharedData];

    int cell_offset_[kNumNeighborCells];
    int cell_nump_[kNumNeighborCells];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];

    //int cell_start_index_[kNumNeighborCells];
    //int cell_end_index_[kNumNeighborCells];
    //int offset_in_cell_;
    //int current_cell_index_;
};





class CKSharedData128
{
public:
    __device__ CKSharedData128(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < 9) {
            int kk = bi * 9 + bj;

            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[kk] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[kk] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9 + bi * 9;
        offset_in_cell_[idx] = 0;

        for (int i = bi * 9; i < bi * 9 + 9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_cf(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 +bi*9<= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
//            position_[idx] = buff_list.position[read_idx];
            c_pressure[idx] = buff_list.correction_pressure[read_idx];
            pha_[idx] = buff_list.phase[read_idx];
            
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < bi*9+9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPos(uint idx) {
        return position_[idx];
    }
    __device__  float & getCor_press(uint idx) {
        return c_pressure[idx];
    }
    __device__  condition & get_pha(uint idx) {
        return pha_[idx];
    }

private:
    //__device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData*2];
    float c_pressure[kNumSharedData * 2];
    condition pha_[kNumSharedData * 2];
    int cell_offset_[9 * 2];
    int cell_nump_[9 * 2];
    int offset_in_cell_[kNumSharedData * 2];
    int current_cell_index_[kNumSharedData * 2];
};






class CMDSharedData128
{
public:
    __device__ CMDSharedData128(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < 9) {
            int kk = bi * 9 + bj;

            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[kk] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[kk] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9+bi*9;
        offset_in_cell_[idx] = 0;

        for (int i = bi*9; i < bi*9+9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_den(float3* pre_pos, condition *pha) {

        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9 +bi*9<= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
            position_[idx] = pre_pos[read_idx];
            pha_[idx] = pha[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9+bi*9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPre_Pos(uint idx) {
        return position_[idx];
    }

private:
    //   __device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData*2];
    int cell_offset_[9 * 2];
    int cell_nump_[9 * 2];
    int offset_in_cell_[kNumSharedData * 2];
    int current_cell_index_[kNumSharedData * 2];
    condition pha_[kNumSharedData * 2];
};

class CMFSharedData128
{
public:
    __device__ CMFSharedData128(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < 9) {

            int kk = bi * 9 + bj;

            ushort3 neighbor_pos = cell_pos + make_ushort3(-1, bj % 3 - 1, bj / 3 % 3 - 1);
            if (neighbor_pos.y < 0 || neighbor_pos.y >= grid_size.y ||
                neighbor_pos.z < 0 || neighbor_pos.z >= grid_size.z) {
                cell_offset_[kk] = 0;
                cell_nump_[kk] = 0;
            }
            else {
                int nid_left, nid_mid, nid_right;
                nid_left = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_mid = CellPos2CellIdx(neighbor_pos, grid_size);
                ++neighbor_pos.x;
                nid_right = CellPos2CellIdx(neighbor_pos, grid_size);
                cell_offset_[kk] =
                    kInvalidCellIdx == nid_left ? cell_offset[nid_mid] : cell_offset[nid_left];
                int my_cell_nump = cell_nump[nid_mid];
                if (kInvalidCellIdx != nid_left) my_cell_nump += cell_nump[nid_left];
                if (kInvalidCellIdx != nid_right) my_cell_nump += cell_nump[nid_right];
                cell_nump_[kk] = my_cell_nump;
            }
        }
        current_cell_index_[idx] = 9+bi*9;
        offset_in_cell_[idx] = 0;

        for (int i = bi*9; i < bi*9+9; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int Read32Data_other_Force(const ParticleBufferList& buff_list) {

        unsigned int idx = threadIdx.x;

        int bj = idx % 32;
        int bi = idx / 32;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (9+bi*9 <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
//            register float3 *tp = (float3*)(buff_list.position_d+read_idx);
 //           position_[idx] = *tp;
//            ev[idx] = buff_list.evaluated_velocity[read_idx];
            //pressure[idx] = buff_list.pressure[read_idx];
            //density[idx] = buff_list.density[read_idx];//1.0f / buff_list.density[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < 9+bi*9) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }

        return num_read;
    }

    __device__  float3 & getPos(uint idx) {
        return position_[idx];
    }
    __device__  float3 & getEV(uint idx) {
        return ev[idx];
    }

private:
    //  __device__ void addOneCellPos(ushort3 &cell_pos);

    float3 position_[kNumSharedData*2];
    float3 ev[kNumSharedData*2];


    int cell_offset_[9*2];
    int cell_nump_[9*2];
    int offset_in_cell_[kNumSharedData*2];
    int current_cell_index_[kNumSharedData*2];
};





class pmfCdapSharedData128
{
public:
    __device__ pmfCdapSharedData128(){}
    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < kNumNeighborCells)
        {

            int kk = bi * 27 + bj;

            cell_offset_[kk] = 0;
            cell_nump_[kk] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(bj % 3 - 1, bj / 3 % 3 - 1, bj / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[kk] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[kk] = cell_offset[neighbor_id];
                cell_nump_[kk] = cell_nump[neighbor_id];
            }
        }
      
        current_cell_index_[idx] = kNumNeighborCells + bi * 27;
        offset_in_cell_[idx] = 0;

        for (int i = bi * 27; i < bi * 27 + kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(ParticleBufferList &buff_list) {
       
        unsigned int idx = threadIdx.x;
        int bj = idx % 32;
        int bi = idx / 32;

        int curr_cell_index = current_cell_index_[idx];
      
        if (bi * 27 + kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
//            positions[idx] = buff_list.position[read_idx];
//            ev[idx] = buff_list.evaluated_velocity[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells + bi * 27) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
    }

    __device__ float3 & getPos(unsigned int idx) {
        return positions[idx];
    }

    __device__ float3 & getEV(unsigned int idx) {
        return ev[idx];
    }

private:
    float3 positions[kNumSharedData * 2];
    float3 ev[kNumSharedData * 2];

    int cell_offset_[kNumNeighborCells * 2];
    int cell_nump_[kNumNeighborCells * 2];
    int offset_in_cell_[kNumSharedData * 2];
    int current_cell_index_[kNumSharedData * 2];
   
};


class pciCdapSharedData128
{
public:
    __device__ pciCdapSharedData128(){}

    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;
        int bi = idx / 32;
        int bj = idx % 32;

        if (bj < kNumNeighborCells)
        {

            int kk = bi * 27 + bj;

            cell_offset_[kk] = 0;
            cell_nump_[kk] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(bj % 3 - 1, bj / 3 % 3 - 1, bj / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[kk] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[kk] = cell_offset[neighbor_id];
                cell_nump_[kk] = cell_nump[neighbor_id];
            }
        }

        current_cell_index_[idx] = kNumNeighborCells;
        offset_in_cell_[idx] = 0;

        for (int i = bi * 27; i < bi * 27 + kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

    __device__ int read32Data(float3 *pos, condition *pha) {

        unsigned int idx = threadIdx.x;
        int bj = idx % 32;
        int bi = idx / 32;

        int curr_cell_index = current_cell_index_[idx];

        if (bi * 27 + kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > bj) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + bj;
            pred_positions[idx] = pos[read_idx];
            phase_[idx] = pha[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells + bi * 27) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
    }

    __device__ float3 & getPre_Pos(unsigned int idx) {
        return pred_positions[idx];
    }

    __device__ condition & getPha(unsigned int idx) {
        return phase_[idx];
    }

private:
    float3 pred_positions[kNumSharedData * 2];
    condition phase_[kNumSharedData * 2];

    int cell_offset_[kNumNeighborCells * 2];
    int cell_nump_[kNumNeighborCells * 2];
    int offset_in_cell_[kNumSharedData * 2];
    int current_cell_index_[kNumSharedData * 2];

    //int cell_start_index_[kNumNeighborCells];
    //int cell_end_index_[kNumNeighborCells];
    //int offset_in_cell_;
    //int current_cell_index_;
};

class pciCdapSharedData
{
public:
	__device__ pciCdapSharedData(){}

    /*__device__ void initialize(int *cell_of, int *cell_nm, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells) {
        cell_end_index_[idx] = 0;
        ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
        int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
        if (kInvalidCellIdx == neighbor_id) {
        cell_start_index_[idx] = kInvalidCellIdx;
        }
        else {
        cell_start_index_[idx] = cell_of[neighbor_id];
        cell_end_index_[idx] = cell_of[neighbor_id] + cell_nm[neighbor_id];
        }
        }

        __syncthreads();

        if (0 == threadIdx.x) {
        current_cell_index_ = kNumNeighborCells;
        offset_in_cell_ = 0;

        for (int i = 0; i < kNumNeighborCells; ++i) {
        if (0 != cell_end_index_[i]) {
        current_cell_index_ = i;
        break;
        }
        }
        }
        }*/
    __device__ void initialize(int *cell_offset, int *cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {//knInitializeSharedMemory2(/*out*/int *sm_cell_offset, int *sm_cell_nump, int &offset_in_cell, int &curr_cell_index,
        //  /*in*/int* cell_offset, int* cell_nump, const ushort3 &cell_pos, const ushort3 &grid_size) {
        unsigned int idx = threadIdx.x;

        if (idx < kNumNeighborCells)
        {
            cell_offset_[idx] = 0;
            cell_nump_[idx] = 0;
            ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
            int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
            if (kInvalidCellIdx == neighbor_id)
            {
                cell_offset_[idx] = kInvalidCellIdx;
            }
            else
            {
                cell_offset_[idx] = cell_offset[neighbor_id];
                cell_nump_[idx] = cell_nump[neighbor_id];
            }
        }
        //        syncthreads1D();

        current_cell_index_[idx] = kNumNeighborCells;
        offset_in_cell_[idx] = 0;

        for (int i = 0; i < kNumNeighborCells; i += 1) {
            if (0 != cell_nump_[i]) {
                current_cell_index_[idx] = i;
                break;
            }
        }
    }

	__device__ int read32Data(float3 *pos, condition *pha) {
        /*unsigned int idx = threadIdx.x;
        int num_read;
        __syncthreads();
        if (kNumNeighborCells <= current_cell_index_) return 0;
        int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
        num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

        if (num_read > idx) {
        int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;
        pred_positions[idx] = pos[read_idx];
        phase_[idx] = pha[read_idx];
        }

        __syncthreads();


        if (0 == idx) {
        if (num_in_cell > num_read + offset_in_cell_) {
        offset_in_cell_ += num_read;
        }
        else {
        int i = current_cell_index_ + 1;
        for (; i < kNumNeighborCells; ++i) {
        if (kInvalidCellIdx != cell_start_index_[i]) break;
        }
        current_cell_index_ = i;
        offset_in_cell_ = 0;
        }
        }
        return num_read;*/
        unsigned int idx = threadIdx.x;

        int curr_cell_index = current_cell_index_[idx];
        // neighbors read complete
        if (kNumNeighborCells <= curr_cell_index) return 0;
        int offset_in_cell = offset_in_cell_[idx];

        int remain_nump = cell_nump_[curr_cell_index] - offset_in_cell;
        int num_read = remain_nump > kNumSharedData ? kNumSharedData : remain_nump;

        if (num_read > idx) {
            int read_idx = cell_offset_[curr_cell_index] + offset_in_cell + idx;
            pred_positions[idx] = pos[read_idx];
            phase_[idx] = pha[read_idx];
        }

        if (remain_nump > kNumSharedData) offset_in_cell_[idx] += kNumSharedData;
        else {
            int next_cell_idx = curr_cell_index + 1;
            while (next_cell_idx < kNumNeighborCells) {
                if (0 != cell_nump_[next_cell_idx]) break;
                ++next_cell_idx;
            }
            current_cell_index_[idx] = next_cell_idx;
            offset_in_cell_[idx] = 0;
        }
        return num_read;
	}

    __device__ float3 & getPre_Pos(unsigned int idx) {
		return pred_positions[idx];
	}

	__device__ condition & getPha(unsigned int idx) {
		return phase_[idx];
	}

private:
	float3 pred_positions[kNumSharedData];
	condition phase_[kNumSharedData];
	
    int cell_offset_[kNumNeighborCells];
    int cell_nump_[kNumNeighborCells];
    int offset_in_cell_[kNumSharedData];
    int current_cell_index_[kNumSharedData];

    //int cell_start_index_[kNumNeighborCells];
    //int cell_end_index_[kNumNeighborCells];
	//int offset_in_cell_;
	//int current_cell_index_;
};

class pciCfSharedData
{
public:
	__device__ pciCfSharedData() {}

	__device__ void initialize(int *cell_of, int *cell_nm, const ushort3 &cell_pos, const ushort3 &grid_size) {
		unsigned int idx = threadIdx.x;

		if (idx < kNumNeighborCells)
		{
			ushort3 neighbor_pos = cell_pos + make_ushort3(idx % 3 - 1, idx / 3 % 3 - 1, idx / 9 % 3 - 1);
			int neighbor_id = CellPos2CellIdx(neighbor_pos, grid_size);
			if (kInvalidCellIdx == neighbor_id)
			{
				cell_start_index_[idx] = kInvalidCellIdx;
			}
			else
			{
				cell_start_index_[idx] = cell_of[neighbor_id];
                cell_end_index_[idx] = cell_of[neighbor_id] + cell_nm[neighbor_id];
			}
		}

		__syncthreads();

		if (0 == threadIdx.x)
		{
			current_cell_index_ = kNumNeighborCells;
			offset_in_cell_ = 0;

			for (int i = 0; i < kNumNeighborCells; ++i)
			{
				if (kInvalidCellIdx != cell_start_index_[i])
				{
					current_cell_index_ = i;
					break;
				}
			}
		}
	}

	__device__ unsigned int read32Data(ParticleBufferList &buff_list) {
		unsigned int idx = threadIdx.x;
		int num_read;

		__syncthreads();

		// neighbors read complete
		if (kNumNeighborCells <= current_cell_index_) return 0;

		// estimate num will read
		int num_in_cell = cell_end_index_[current_cell_index_] - cell_start_index_[current_cell_index_];
		num_read = (num_in_cell - offset_in_cell_) > kNumSharedData ? kNumSharedData : (num_in_cell - offset_in_cell_);

		if (num_read > idx)
		{
			// get index
			int read_idx = cell_start_index_[current_cell_index_] + offset_in_cell_ + idx;

			// read data
//			position_[idx] = buff_list.position[read_idx];
			//ev_[idx] = buff_list.evaluated_velocity[read_idx];
			correction_pressure_[idx] = buff_list.correction_pressure[read_idx];
			//density_[idx] = buff_list.density[read_idx];
			phase_[idx] = buff_list.phase[read_idx];
		}

		__syncthreads();

		// update info
		if (0 == idx)
		{
			// this cell has not been completed
			if (num_in_cell > num_read + offset_in_cell_)
			{
				offset_in_cell_ += num_read;
			}
			else // new cell
			{
				int i = current_cell_index_ + 1;
				for (; i < kNumNeighborCells; ++i)
				{
					if (kInvalidCellIdx != cell_start_index_[i])
					{
						break;
					}
				}
				current_cell_index_ = i;
				offset_in_cell_ = 0;
			}
		}

		return num_read;
	}

	__device__ float3 & getPosition(unsigned int idx) { return position_[idx]; }

	//__device__ float3 & getEV(unsigned int idx) { return ev_[idx]; }

	__device__ float & getPressure(unsigned int idx) { return correction_pressure_[idx]; }

	//__device__ float & getDensity(unsigned int idx) { return density_[idx]; }

	__device__ condition & getPhase(unsigned int idx) { return phase_[idx]; }

private:
	float3  position_[kNumSharedData];
	//float3  ev_[kNumSharedData];
	float   correction_pressure_[kNumSharedData];
	//float   density_[kNumSharedData];
	condition phase_[kNumSharedData];

	int cell_start_index_[kNumNeighborCells];
	int cell_end_index_[kNumNeighborCells];

	int offset_in_cell_;
	int current_cell_index_;
};
//sf class PCISPH over-------------------------------------------------------------------

}

#endif/*_SOLVER_KERNEL_COMMON_CUH_*/
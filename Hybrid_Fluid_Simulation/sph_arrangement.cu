//
// sph_arrangement.cu
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#include "sph_arrangement.cuh"
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "cuda_prescan/scan.cuh"
#include "gpu_model.cuh"
#include "sph_utils.cuh"

#include<fstream>

namespace sph
{


/****************************GPU_COUNT_SORT**************************/

__global__ void CountingSort_Cell_Sum(int *p_offset, int *hashId, int *cell_numbers, int iSize, float4 *position, float cell_size, ushort3 grid_size)
{
    int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

    if (x_id < iSize)
    {
        int hid = ParticlePos2CellIdx(position[x_id], grid_size, cell_size);
        hashId[x_id] = hid;
        p_offset[x_id] = atomicAdd(cell_numbers + hid, 1);
    }
}
__global__ void CountingSort_Cell_Sum_two(int *p_offset, int *hashId, int *cell_numbers, int iSize, int *block_reqs, int numc)
{
    int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (x_id < iSize)
    {
        int selfHash = hashId[x_id];
//        selfHash = block_reqs[selfHash] > 0 ? (selfHash + numc) : selfHash;
        

        selfHash = block_reqs[selfHash] * 32 > p_offset[x_id] ? (selfHash + numc) : selfHash;
     /*   int thd = block_reqs[selfHash] * 32;
        if (thd <= p_offset[x_id]){
            selfHash = selfHash;
        }
        else{
            selfHash = (selfHash + numc);
        }*/
        hashId[x_id] = selfHash;
        p_offset[x_id] = atomicAdd(cell_numbers + selfHash, 1);
    }
}
__global__ void CountingSort_Cell_Sum_two9(int *p_offset, int *hashId, int *cell_numbers, int iSize, float4 *position, float cell_size, ushort3 grid_size,int *block_reqs, int numc)
{
    int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (x_id < iSize)
    {
        int selfHash = ParticlePos2CellIdx(position[x_id], grid_size, cell_size);
        selfHash = block_reqs[selfHash] > 0 ? (selfHash + numc) : selfHash;
        hashId[x_id] = selfHash;
        p_offset[x_id] = atomicAdd(cell_numbers + selfHash, 1);
    }
}
__global__
void CountingSort_Offset(int *cell_offset, int cellNum, int iSize)
{
    int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    x_id++;
    if (x_id <= cellNum)
    {
        int P = x_id & (iSize - 1);
        if (0 == P)
            P = iSize;
        if (P > (iSize >> 1))
        {
            x_id--;
            cell_offset[x_id] = cell_offset[x_id] + cell_offset[x_id + (iSize >> 1) - P];
        }
    }
}

__global__ void CountingSort_Cell_Sum_two_M(int* hashp, int *p_offset, int *hashId, int *cell_numbers, int iSize, int *block_reqs, int numc)
{
	int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (x_id < iSize)
	{
		int selfHash = (hashId[x_id] >> 6);
		//selfHash = block_reqs[selfHash] > 0 ? (selfHash + numc) : selfHash;


		selfHash = block_reqs[selfHash] <<5 > p_offset[x_id] ? (selfHash + numc) : selfHash;
		/*   int thd = block_reqs[selfHash] * 32;
		if (thd <= p_offset[x_id]){
		selfHash = selfHash;
		}
		else{
		selfHash = (selfHash + numc);
		}*/
		hashp[x_id] = selfHash;
		p_offset[x_id] = atomicAdd(cell_numbers + selfHash, 1);
	}
}
__global__ void clean_data(int *cell_Numx, int numc)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < numc)
    {
        cell_Numx[id] = 0;
    }
}

__global__ void clean_data_two(int *cell_Numx, int numc, int* start_i, int* end_i)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < numc)
    {
        cell_Numx[id] = 0;
        if (id < (numc >> 1)){
            start_i[id] = -1;
            end_i[id] = 0;
        }
    }
}

void CountingSort_Offest_P(int block_cell, int cell_s, int *cell_offset, int cellNum)
{
    int iSize = 1;
    while (iSize < cellNum)
    {
        iSize <<= 1;
        CountingSort_Offset << <block_cell, cell_s >> >(cell_offset, cellNum, iSize);
    }
}
__global__ void CountingSort_Result9(int *p_offset, int *hash, int *hash_new, int *cell_offset, int *index, int* index_new, int num, ParticleBufferList old_data, ParticleBufferList new_data)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < num)
    {
        int x_id = hash[id];
        int iStart = cell_offset[x_id];
        int p_id = iStart + p_offset[id];
        hash_new[p_id] = x_id;
        new_data.position_d[p_id] = old_data.position_d[id];
        new_data.velocity[p_id] = old_data.velocity[id];
        new_data.evaluated_velocity[p_id] = old_data.evaluated_velocity[id];
        //new_data.color[p_id] = old_data.color[id];
        new_data.phase[p_id] = old_data.phase[id];
    }
}
__global__ void CountingSort_Result(int *p_offset_p, int *p_offset, int *hash, int *hash_new, int *cell_offset, int num, ParticleBufferList old_data, ParticleBufferList new_data)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < num)
    {
        int x_id = hash[id];
        int iStart = cell_offset[x_id];
        int p_id = iStart + p_offset[id];
        hash_new[p_id] = x_id;

        p_offset_p[p_id] = p_offset[id];

        new_data.position_d[p_id] = old_data.position_d[id];
        new_data.velocity[p_id] = old_data.velocity[id];
        new_data.evaluated_velocity[p_id] = old_data.evaluated_velocity[id];
        new_data.color[p_id] = old_data.color[id];
//        new_data.phase[p_id] = old_data.phase[id];
    }
}
__global__ void CountingSort_Result_two(int *p_offset, int *hash, int *hash_new, int *cell_offset,int num, ParticleBufferList old_data, ParticleBufferList new_data)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < num)
    {
        int iStart = 0;
        int x_id = hash[id];
        if (0 != x_id)
            iStart = cell_offset[x_id - 1];
        int p_id = iStart + p_offset[id];

        //index_new[p_id] = index[id];
        hash_new[p_id] = x_id;
        new_data.position_d[p_id] = old_data.position_d[id];
        new_data.velocity[p_id] = old_data.velocity[id];
        new_data.evaluated_velocity[p_id] = old_data.evaluated_velocity[id];
        new_data.phase[p_id] = old_data.phase[id];
    }
}
__global__ void CountingSort_Result_two9(int *p_offset, int *hash, int *hash_new, int *cell_offset, int *index,int num)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < num)
    {
        int iStart = 0;
        int x_id = hash[id];
        if (0 != x_id)
            iStart = cell_offset[x_id - 1];
        int p_id = iStart + p_offset[id];
        index[p_id] = id;
        hash_new[p_id] = x_id;
    }
}
__global__ void copy_data(int *des_hash, int *des_index, int *src_hash, int *src_index, int numbers)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < numbers)
    {
        des_hash[id] = src_hash[id];
        des_index[id] = src_index[id];
    }
}
__global__ void Get_offset_and_num(int *cell_type, int *pcell_num_two, int *pd_cell_offset_, int *pd_cell_num_, int numc) //(int *des_hash, int *des_index, int *src_hash, int *src_index, int numbers)
{
    int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
    if (id < numc)
    {
        if (id == 0) {
            if (cell_type[id] == 0){
                pd_cell_offset_[id] = 0;
                pd_cell_num_[id] = 0;
                return;
            }
            if (cell_type[id] == 1) {
                pd_cell_offset_[id] = 0;
                pd_cell_num_[id] = pcell_num_two[id];
                return;
            }
            if (cell_type[id] == 2){
                pd_cell_offset_[id] = pcell_num_two[id+numc-1];
                pd_cell_num_[id] = pcell_num_two[id + numc] - pcell_num_two[id + numc - 1];
                return;
            }
        }
        else{
            if (cell_type[id] == 0){
                pd_cell_offset_[id] = pcell_num_two[id - 1];
                pd_cell_num_[id] = 0;
                return;
            }
            if (cell_type[id] == 1) {
                pd_cell_offset_[id] = pcell_num_two[id - 1];
                pd_cell_num_[id] = pcell_num_two[id] - pcell_num_two[id - 1];
                return;
            }
            if (cell_type[id] == 2){
                pd_cell_offset_[id] = pcell_num_two[id + numc - 1];
                pd_cell_num_[id] = pcell_num_two[id + numc] - pcell_num_two[id + numc - 1];
                return;
            }
        }
    }
}
void Arrangement::CountingSortCUDA()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numc_+1, num_thread);

    clean_data << <num_blockc, num_thread >> >(d_cell_nump_, numc_);
    CountingSort_Cell_Sum << <num_block, num_thread >> >(d_p_offset_, d_hash_, d_cell_nump_, nump_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_);
    cudaMemcpy(d_cell_offset_+ 1, d_cell_nump_, sizeof(int) * numc_, cudaMemcpyDeviceToDevice);
    CountingSort_Offest_P(num_blockc, num_thread, d_cell_offset_, numc_+1);
    CountingSort_Result << <num_block, num_thread >> >(d_p_offset_p,d_p_offset_, d_hash_, hashp, d_cell_offset_, nump_, buff_list_.get_buff_list(), buff_temp_.get_buff_list());

    //copy_data << <num_block, num_thread >> >(d_hash_, d_index_, hashp, indexp, nump_);
    //cudaMemcpy(d_hash_, hashp, sizeof(int) * nump_, cudaMemcpyDeviceToDevice);

    int *p = d_hash_;
    d_hash_ = hashp;
    hashp = p;
    buff_list_.swapObj(buff_temp_);

    /*unsigned int nc = numc_;
    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *ha = new int[nump_];
    int *cnp = new int[nc];
    cudaMemcpy(ha, buff_temp_.get_buff_list().position, nump_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnp, d_cell_offset_, nc*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nc; i++){
        out << cnp[i] << std::endl;;
    }
    for (int i = 0; i < nump_; i++){
        out << ha[i] << std::endl;;
    }
    std::cout << middle_value_;
    std::cout << "---------------- ";*/

   /* unsigned int nc = numc_;
    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *ha = new int[nump_];
    int *cof = new int[nc];
    int *cnp = new int[nc];
    cudaMemcpy(ha, d_hash_, nump_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cof, d_cell_nump_, nc*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnp, d_cell_offset_, nc*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nc; i++){
       
        out << cnp[i] << "            " << cof[i] << std::endl;;
    }
    std::cout << middle_value_;
    std::cout << "---------------- ";*/
    /*unsigned int nc = numc_;

    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *cof = new int[nc];
    int *cnp = new int[nc];

    cudaMemcpy(cof, d_cell_nump_, nc*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnp, d_cell_offset_, nc*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nc; i++){
        out << cnp[i] << std::endl;
    }
    std::cout << "---------------- ";*/
}
__device__
inline int hash2cellid(int base, int num) { return base % num; }
__device__
inline int hash2blockeq(int base, int num) { return base / num; }
__global__
void knFindCellRangeAndHybridModeMiddleValue(int numc, int *start_idx, int *end_idx, int *mid_val, int *hash, unsigned int nump)
{
    extern __shared__ int shared_hash[];
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    int self_hash;
    if (idx < nump){
        self_hash = hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;
        if (idx > 0 && threadIdx.x == 0){
            shared_hash[0] = hash[idx - 1];
        }
    }
    __syncthreads();
    if (idx < nump) {
        int prior_hash = idx == 0 ? -1 : shared_hash[threadIdx.x];
        if (self_hash != prior_hash){
            start_idx[hash2cellid(self_hash, numc)] = idx;
            if (idx > 0){
                end_idx[hash2cellid(shared_hash[threadIdx.x], numc)] = idx;
            }
            if (hash2blockeq(self_hash, numc) > 0) {
                if (idx == 0 || hash2blockeq(prior_hash, numc) == 0){
                    *mid_val = idx;
                }
            }
        }
        if (idx == nump - 1){
            end_idx[hash2cellid(self_hash, numc)] = idx + 1;
        }
    }
}
__global__
void knFindHybridModeMiddleValue(int numc, int *mid_val, int *hash, unsigned int nump)
{
    extern __shared__ int shared_hash[];
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    if (idx == 0){ *mid_val = -1; }
    int self_hash;
    if (idx < nump){
        self_hash = hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;
        if (idx > 0 && threadIdx.x == 0){
            shared_hash[0] = hash[idx - 1];
        }
    }
    __syncthreads();
    if (idx < nump) {
        int prior_hash = idx == 0 ? -1 : shared_hash[threadIdx.x];
        if (self_hash != prior_hash){
            if (hash2blockeq(self_hash, numc) > 0) {
                if (idx == 0 || hash2blockeq(prior_hash, numc) == 0){
                    *mid_val = idx;
                }
            }
        }
    }
}


__global__
void get_num_offset(int *start_i, int *end_i, int *cell_of, int* cell_num, unsigned int numc)
{
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= numc) return;
    if (start_i[idx] > -1){
        cell_of[idx] = start_i[idx];
        cell_num[idx] = end_i[idx] - start_i[idx];
    }
    else{
        cell_num[idx] = 0;
        //cell_of[idx] = start_i[idx];
    }
}


void Arrangement::CountingSortCUDA_Two()
{
    int numCell = numc_ << 1;
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numCell, num_thread);


    //clean_data_two<<<num_blockc, num_thread >>>(cell_num_two, numCell, d_start_index_, d_end_index_);
    clean_data << <num_blockc, num_thread >> >(cell_num_two, numCell);

    CountingSort_Cell_Sum_two <<<num_block, num_thread >>>(d_p_offset_, d_hash_, cell_num_two, nump_, d_block_reqs_, numc_);

    CountingSort_Offest_P(num_blockc, num_thread, cell_num_two, numCell);

    CountingSort_Result_two <<<num_block, num_thread >>>(d_p_offset_, d_hash_, hashp, cell_num_two, nump_, buff_list_.get_buff_list(), buff_temp_.get_buff_list());

    Get_offset_and_num << <ceil_int(numc_, num_thread), num_thread >> >(cell_type, cell_num_two, d_cell_offset_, d_cell_nump_, numc_);
    unsigned int shared_mem_size = (num_thread + 1) * sizeof(int);

    /*int nc = numc_;
    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *cof = new int[nc];
    int *cnp = new int[nc];
    cudaMemcpy(cof, d_cell_offset_, nc*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(cnp, cell_nump, nc*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nc; i++){
        out << cof[i] << ' ' << std::endl;//<< cnp[i];
    }
    std::cout << "---------------- ";*/


    int *p = d_hash_;
    d_hash_ = hashp;
    hashp = p;


    knFindHybridModeMiddleValue << <num_block, num_thread, shared_mem_size >> >(numc_, d_middle_value_, d_hash_, nump_);
    //CUDA_SAFE_CALL(cudaMemset(d_middle_value_, 0xffffffff, sizeof(int)));
    //knFindCellRangeAndHybridModeMiddleValue <<<num_block, num_thread, shared_mem_size >>>(numc_, d_start_index_, d_end_index_, d_middle_value_, d_hash_, nump_);
    //int bck = ceil_int(numc_, num_thread);
    //get_num_offset << <bck, num_thread >> >(d_start_index_, d_end_index_, d_cell_offset_, d_cell_nump_, numc_);


    CUDA_SAFE_CALL(cudaMemcpy(&middle_value_, d_middle_value_, sizeof(int), cudaMemcpyDeviceToHost));

   // std::cout << "asdfas9999999999999999999999999999999999999999999" << std::endl;

    buff_list_.swapObj(buff_temp_);
}


/**********************************************************************************************************/


/****************************** Kernel ******************************/

#define HASH2CELLIDX(X) (X & 0x3fffffff)
#define HASH2BLOCKREQ(X) (X >> 30)

__global__
void knCalculateHash(int *hash, int *index, float4 *position, float cell_size, ushort3 grid_size, unsigned int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= nump) return;

    hash[idx] = ParticlePos2CellIdx(position[idx], grid_size, cell_size);
    index[idx] = idx;
}

__global__
void knCalculateHashWithBlockReq(int numc, int *hash, int *index, int *block_reqs, unsigned int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    if (idx >= nump) return;
    int selfHash = hash[idx];
    hash[idx] = block_reqs[selfHash] > 0 ? (selfHash + numc) : selfHash;
    index[idx] = idx;
}

__global__
void knReindexParticles(ParticleBufferList old_data, ParticleBufferList new_data, int *index, unsigned int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx < nump)
    {
        int sorted_idx = index[idx];

        new_data.position_d[idx] = old_data.position_d[sorted_idx];
        new_data.velocity[idx] = old_data.velocity[sorted_idx];
        new_data.evaluated_velocity[idx] = old_data.evaluated_velocity[sorted_idx];
        new_data.color[idx] = old_data.color[sorted_idx];

		//sf add
		new_data.phase[idx] = old_data.phase[sorted_idx];
    }
}

__global__
void knFindCellRange(int *start_idx, int *end_idx, int *hash, unsigned int nump)
{
    extern __shared__ int shared_hash[];
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    int self_hash;

    if (idx < nump)
    {
        self_hash = hash[idx];
        shared_hash[threadIdx.x + 1] = self_hash;

        if (idx > 0 && threadIdx.x == 0)
        {
            shared_hash[0] = hash[idx - 1];
        }
    }

    __syncthreads();

    if (idx < nump)
    {
        if (idx == 0 || self_hash != shared_hash[threadIdx.x])
        {
            start_idx[self_hash] = idx;

            if (idx > 0)
            {
                end_idx[shared_hash[threadIdx.x]] = idx;
            }
        }

        if (idx == nump - 1)
        {
            end_idx[self_hash] = idx + 1;
        }
    }
}






//__global__
//void knArrangeBlockTasks(BlockTask *block_tasks, int *num_block, int *block_reqs, int *breqs_offset, ushort3 grid_size, int numc)
//{
//    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
//
//    if (idx >= numc) return;
//
//    int offset = breqs_offset[idx];
//    int numb = block_reqs[idx];
//
//    for (int i = offset; i < offset + numb; ++i)
//    {
//        BlockTask bt;
//        bt.cell_pos = CellIdx2CellPos(idx, grid_size);
//        bt.sub_idx = i - offset;
//        block_tasks[i] = bt;
//    }
//
//    if (idx == numc - 1) *num_block = offset + numb;
//}

__global__
void countCellNum(int *hash, float4 *position, float cell_size, ushort3 grid_size, int *cell_num, unsigned int nump)
{
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
    if (idx >= nump) return;
    int hav = ParticlePos2CellIdx(position[idx], grid_size, cell_size);
    hash[idx] = hav;
    atomicAdd(cell_num + hav, 1);
}




//__global__ void countCellNum(int *hashId, int *cell_numbers, int iSize, float3 *position, float cell_size, ushort3 grid_size)
//{
//    unsigned int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
//
//    if (x_id < iSize)
//    {
//        int hid = ParticlePos2CellIdx(position[x_id], grid_size, cell_size);
//        hashId[x_id] = hid;
//        atomicAdd(cell_numbers + hid, 1);
//    }
//}







/****************************** Arrangement ******************************/

Arrangement::Arrangement(ParticleBufferObject &buff_list, ParticleBufferObject &buff_temp,unsigned int nump, unsigned int nump_capacity, float cell_size, ushort3 grid_size)
    : buff_list_(buff_list), buff_temp_(buff_temp),  nump_(nump), nump_capacity_(nump_capacity), cell_size_(cell_size), grid_size_(grid_size)
{
    numc_ = grid_size.x * grid_size.y * grid_size.z;

    //CUDA_SAFE_CALL(cudaMalloc(&d_start_index_, numc_ * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc(&d_end_index_, numc_ * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_hash_, nump_capacity * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&d_index_, nump_capacity * sizeof(int)));
    // SMS
	CUDA_SAFE_CALL(cudaMalloc(&d_hash_p, nump_capacity * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&hashp, nump_ * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc(&indexp, nump_ * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_block_reqs_, numc_ * sizeof(int)));


    CUDA_SAFE_CALL(cudaMalloc(&d_task_array_offset_32_, numc_ * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_breqs_offset_, numc_ * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc(&cell_num_, numc_ * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&cell_num_two, (2 * numc_+1) * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc(&cell_type, numc_ * sizeof(int)));


    CUDA_SAFE_CALL(cudaMalloc(&d_block_task_, numc_ * 10 * sizeof(BlockTask)));
    CUDA_SAFE_CALL(cudaMalloc(&d_num_block_, sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_middle_value_, sizeof(int)));


    CUDA_SAFE_CALL(cudaMalloc(&d_num_cta_, sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_cell_offset_, (numc_+1) * sizeof(int)));

    //CUDA_SAFE_CALL(cudaMalloc(&d_cell_offset_data, (numc_ + 1) * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_cell_nump_, numc_ * sizeof(int)));


    CUDA_SAFE_CALL(cudaMalloc(&d_p_offset_, nump_ * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc(&d_p_offset_p, nump_ * sizeof(int)));

	CUDA_SAFE_CALL(cudaMalloc(&d_cell_offset_M, (numc_ * 64 + 1) * sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&d_cell_nump_M, (numc_ * 64 + 1) * sizeof(int)));

    preallocBlockSumsInt(numc_);

    CUDA_SAFE_CALL(cudaMemset(d_num_cta_, 0, sizeof(int)));

    gpu_model::allocateGPUModel(p_gpu_model_);
}

Arrangement::~Arrangement()
{
    CUDA_SAFE_CALL(cudaFree(cell_type));
    CUDA_SAFE_CALL(cudaFree(d_task_array_offset_32_));
    CUDA_SAFE_CALL(cudaFree(d_p_offset_));


    CUDA_SAFE_CALL(cudaFree(d_p_offset_p));

    CUDA_SAFE_CALL(cudaFree(hashp));
    //CUDA_SAFE_CALL(cudaFree(indexp));
    //CUDA_SAFE_CALL(cudaFree(d_start_index_));
    //CUDA_SAFE_CALL(cudaFree(d_end_index_));
    CUDA_SAFE_CALL(cudaFree(d_hash_));
	CUDA_SAFE_CALL(cudaFree(d_hash_p));
    CUDA_SAFE_CALL(cudaFree(d_index_));
    CUDA_SAFE_CALL(cudaFree(d_block_reqs_));
    CUDA_SAFE_CALL(cudaFree(d_breqs_offset_));
    CUDA_SAFE_CALL(cudaFree(d_block_task_));
    CUDA_SAFE_CALL(cudaFree(d_num_block_));
    CUDA_SAFE_CALL(cudaFree(d_middle_value_));
    //CUDA_SAFE_CALL(cudaFree(cell_num_));
    CUDA_SAFE_CALL(cudaFree(cell_num_two));

    CUDA_SAFE_CALL(cudaFree(d_num_cta_));
    //CUDA_SAFE_CALL(cudaFree(d_cell_offset_data));
    CUDA_SAFE_CALL(cudaFree(d_cell_offset_));
    CUDA_SAFE_CALL(cudaFree(d_cell_nump_));
	CUDA_SAFE_CALL(cudaFree(d_cell_offset_M));
	CUDA_SAFE_CALL(cudaFree(d_cell_nump_M));

    deallocBlockSumsInt();

    gpu_model::freeGPUModel(p_gpu_model_);
}

int Arrangement::arrangeTRAMode()
{
    calculateHash();
    //sortIndexByHash();
    CountingSortCUDA();
    reindexParticles();
    findCellRange();

    buff_list_.swapObj(buff_temp_);

    return nump_;
}


__global__
void knInsertParticles(int *cell_nump, int* p_offset, float4 *position, float cell_size, ushort3 grid_size, unsigned int nump) {
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= nump) return;

    int cell_id = ParticlePos2CellIdx(position[idx], grid_size, cell_size);
    p_offset[idx] = atomicAdd(cell_nump + cell_id, 1);
}

__global__
void s_clean_data(int *cell_nump, unsigned int nump) {
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= nump) return;

    cell_nump[idx] = 0;
}

void Arrangement::CSInsertParticles() {
    int num_thread = 128;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numc_, num_thread);
    s_clean_data << <num_blockc, num_thread >> >(d_cell_nump_, numc_);
    //CUDA_SAFE_CALL(cudaMemset(d_cell_nump_, 0, numc_ * sizeof(int)));
    knInsertParticles << <num_block, num_thread >> >(d_cell_nump_, d_p_offset_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_, nump_);
}

__global__
void knCountingSortFull(ParticleBufferList old_data, ParticleBufferList new_data,
int *cell_offset, int *p_offset, float cell_size, ushort3 grid_size, unsigned int nump) {
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= nump) return;

    int cell_id = ParticlePos2CellIdx(old_data.position_d[idx], grid_size, cell_size);
    int sorted_idx = cell_offset[cell_id] + p_offset[idx];

    new_data.position_d[sorted_idx] = old_data.position_d[idx];
    new_data.velocity[sorted_idx] = old_data.velocity[idx];
    new_data.evaluated_velocity[sorted_idx] = old_data.evaluated_velocity[idx];
}

void Arrangement::CSCountingSortFull() {
    int num_thread = 128;
    int num_block = ceil_int(nump_, num_thread);

    knCountingSortFull << <num_block, num_thread >> >(buff_list_.get_buff_list(), buff_temp_.get_buff_list(), d_cell_offset_, d_p_offset_, cell_size_, grid_size_, nump_);
}

void Arrangement::sortParticles() {



    CSInsertParticles();
    prescanArrayRecursiveInt(d_cell_offset_, d_cell_nump_, numc_, 0);
    CSCountingSortFull();

    buff_list_.swapObj(buff_temp_);
}


__global__
void knCalculateRequiredCTAs(int *cta_offset, int *cta_reqs, int *cell_nump, int cta_size, unsigned int numc) {
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= numc) return;
    register int nm = ceil_int(cell_nump[idx], cta_size);
    cta_reqs[idx] = nm;
    cta_offset[idx + 1] = nm;
}


void Arrangement::CSCalculateRequiredCTAsFixed(int *cta_offset, int* d_cta_reqs, int cta_size) {
    int num_thread = 256;
    int num_block = ceil_int(numc_, num_thread);

    knCalculateRequiredCTAs << <num_block, num_thread >> >(cta_offset, d_cta_reqs, d_cell_nump_, cta_size, numc_);
}
__global__
void knArrangeTasksFixed(BlockTask *block_tasks, int *num_block, int *block_reqs, int *breqs_offset, ushort3 grid_size, int cta_size, int numc) {
    unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

    if (idx >= numc) return;

    int offset = breqs_offset[idx];
    int numb = block_reqs[idx];
    int p_offset = 0;
    ushort3 cell_pos = CellIdx2CellPos(idx, grid_size);

    for (int i = offset; i < offset + numb; ++i) {
        BlockTask bt;
//        bt.cell_pos = cell_pos;
        bt.p_offset = p_offset;
		bt.cellid = idx;
        block_tasks[i] = bt;
        p_offset += cta_size;
    }

    if (idx == numc - 1) {
        *num_block = offset + numb;
        //num_block[0] = num_block[1] = num_block[2] = 0;
        /*switch (cta_size) {
        case 32: num_block[0] = offset + numb; break;
        case 96: num_block[1] = offset + numb; break;
        case 288: num_block[2] = offset + numb; break;
        default:break;
        }*/
    }
}
__global__
void knArrangeTasksFixedM(int *hash, int* celloff, int *cellnum, BlockTask *block_tasks, int *num_block, int *block_reqs, int *breqs_offset, ushort3 grid_size, int cta_size, int numc) {
	unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);

	if (idx >= numc) return;
	int start = celloff[idx];
	int offset = breqs_offset[idx];
	int numb = block_reqs[idx];
	int p_offset = 0;
	int nump = cellnum[idx];
	//ushort3 cell_pos = CellIdx2CellPos(idx, grid_size);

	for (int i = offset; i < offset + numb; ++i) {
		int hashA = hash[start + p_offset];
		int xxi = ((hashA & 0x030) >> 4);
		int zzi = ((hashA & 0x0c) >> 2);
		BlockTask bt;
	//	bt.cell_pos = cell_pos;
		bt.p_offset = p_offset;
		bt.cellid = idx;
		//block_tasks[i] = bt;
		p_offset += cta_size;
		int  hashB;// = hash[start + p_offset - 1];
		if (p_offset >= nump){
			hashB = hash[start + nump - 1];

		}
		else{
			hashB = hash[start + p_offset - 1];
		}
		int xxx = ((hashB & 0x030) >> 4);
		int zzz = ((hashB & 0x0c) >> 2);
		//bt.isSame &= 0x0;
		bt.xxi = xxi;
		bt.xxx = xxx;

		if (xxi == xxx){
			bt.zzi = zzi;
			bt.zzz = zzz;
		}
		else{
			bt.zzi = 0;
			bt.zzz = 3;
		}
		block_tasks[i] = bt;
	}

	if (idx == numc - 1) {
		*num_block = offset + numb;
	}
}

__global__
void judgeTask(BlockTask *block_tasks, int *num_block) {
	unsigned int idx = threadIdx.x + __umul24(blockDim.x, blockIdx.x);
	int numb = num_block[0];
	if (idx >= numb) return;
	if (numb % 2 == 0){
		if (idx % 2 == 0){
			if (block_tasks[idx].cellid == block_tasks[idx + 1].cellid){
				block_tasks[idx].isSame = 1;
				block_tasks[idx + 1].isSame = 1;
				if (block_tasks[idx].xxi == block_tasks[idx + 1].xxx){
					block_tasks[idx].zzz = block_tasks[idx + 1].zzz;
				}
				else{
					block_tasks[idx].zzi = 0;// block_tasks[idx + 1].zzz;
					block_tasks[idx].zzz = 3;
				}
				block_tasks[idx].xxx = block_tasks[idx + 1].xxx;
			}
			else{
				block_tasks[idx].isSame = 0;
				block_tasks[idx + 1].isSame = 0;
			}
		}
	}
	else{
		if (idx % 2 == 0 && idx<numb - 1){
			if (block_tasks[idx].cellid == block_tasks[idx + 1].cellid){
				block_tasks[idx].isSame = 1;
				block_tasks[idx + 1].isSame = 1;
				if (block_tasks[idx].xxi == block_tasks[idx + 1].xxx){
					block_tasks[idx].zzz = block_tasks[idx + 1].zzz;
				}
				else{
					block_tasks[idx].zzi = 0;// block_tasks[idx + 1].zzz;
					block_tasks[idx].zzz = 3;
				}
				block_tasks[idx].xxx = block_tasks[idx + 1].xxx;
			}
			else{
				block_tasks[idx].isSame = 0;
				block_tasks[idx + 1].isSame = 0;
			}
		}
		if (idx == numb - 1){
			block_tasks[idx].isSame = 1;
			block_tasks[idx + 1].isSame = 1;
			block_tasks[idx + 1].cellid = block_tasks[idx].cellid;
			block_tasks[idx + 1].p_offset = block_tasks[idx].p_offset + 32;
		}
	}
}

void Arrangement::arrangeBlockTasksFixedM(int *hash, int *celloff, int *cellnum, BlockTask* d_task_array, int* d_cta_reqs, int* d_task_array_offset, int cta_size) {
	int num_thread = 128;
	int num_block = ceil_int(numc_, num_thread);

	knArrangeTasksFixedM << <num_block, num_thread >> >(hash, celloff, cellnum, d_task_array, d_num_cta_, d_cta_reqs, d_task_array_offset, grid_size_, cta_size, numc_);

	CUDA_SAFE_CALL(cudaMemcpy(&h_num_cta_, d_num_cta_, sizeof(int), cudaMemcpyDeviceToHost));

	judgeTask << <ceil_int(h_num_cta_, num_thread), num_thread >> >(d_task_array, d_num_cta_);
}

void Arrangement::arrangeBlockTasksFixed(BlockTask* d_task_array, int* d_cta_reqs, int* d_task_array_offset, int cta_size) {
    int num_thread = 128;
    int num_block = ceil_int(numc_, num_thread);

    knArrangeTasksFixed << <num_block, num_thread >> >(d_task_array, d_num_cta_, d_cta_reqs, d_task_array_offset, grid_size_, cta_size, numc_);

    CUDA_SAFE_CALL(cudaMemcpy(&h_num_cta_, d_num_cta_, sizeof(int), cudaMemcpyDeviceToHost));

	judgeTask << <ceil_int(h_num_cta_, num_thread), num_thread >> >(d_task_array, d_num_cta_);

	//BlockTask *h_task = new BlockTask[h_num_cta_[0]];
	//std::ofstream outt("hkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhkhk.txt");
	//CUDA_SAFE_CALL(cudaMemcpy(h_task, d_task_array, h_num_cta_[0] * sizeof(BlockTask), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < h_num_cta_[0]; i++){
	//	if (i % 2 == 0){
	//		if (h_task[i].isSame != h_task[i + 1].isSame)
	//			outt << 404 << std::endl;
	//		//outt << h_task[i].cell_pos.x << "     " << h_task[i].cell_pos.y << "     " << h_task[i].cell_pos.z << "     " << h_task[i].isSame << "     " << h_task[i].p_offset << std::endl;
	//	}
	//}
//	std::cout << "asdfasdf";

}
void Arrangement::assignTasksFixedCTA() {
    int num_thread = 128;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numc_ + 1, num_thread);

    clean_data << <num_blockc, num_thread >> >(d_cell_nump_, numc_);
    CountingSort_Cell_Sum << <num_block, num_thread >> >(d_p_offset_, d_hash_, d_cell_nump_, nump_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_);
    cudaMemcpy(d_cell_offset_ + 1, d_cell_nump_, sizeof(int) * numc_, cudaMemcpyDeviceToDevice);
    CountingSort_Offest_P(num_blockc, num_thread, d_cell_offset_, numc_ + 1);
    CountingSort_Result << <num_block, num_thread >> >(d_p_offset_p, d_p_offset_, d_hash_, hashp, d_cell_offset_, nump_, buff_list_.get_buff_list(), buff_temp_.get_buff_list());

    //copy_data << <num_block, num_thread >> >(d_hash_, d_index_, hashp, indexp, nump_);
    //cudaMemcpy(d_hash_, hashp, sizeof(int) * nump_, cudaMemcpyDeviceToDevice);

    int *p = d_hash_;
    d_hash_ = hashp;
    hashp = p;
    buff_list_.swapObj(buff_temp_);


    knCalculateRequiredCTAs << <num_blockc, num_thread >> >(d_task_array_offset_32_, d_block_reqs_, d_cell_nump_, 32, numc_);
    CountingSort_Offest_P(num_blockc, num_thread, d_task_array_offset_32_, numc_ + 1);
//        prescanArrayRecursiveInt(d_task_array_offset_32_, d_block_reqs_, numc_, 0);
    knArrangeTasksFixed << <num_blockc, num_thread >> >(d_block_task_, d_num_cta_, d_block_reqs_, d_task_array_offset_32_, grid_size_, 32, numc_);

    CUDA_SAFE_CALL(cudaMemcpy(&h_num_cta_, d_num_cta_, sizeof(int), cudaMemcpyDeviceToHost));
	judgeTask << <ceil_int(h_num_cta_, num_thread), num_thread >> >(d_block_task_, d_num_cta_);
}












void Arrangement::arrangeSMSMode()
{
    //cudaEvent_t start, end;
    //CUDA_SAFE_CALL(cudaEventCreate(&start));
    //CUDA_SAFE_CALL(cudaEventCreate(&end));
    //CUDA_SAFE_CALL(cudaEventRecord(start));

    calculateHash();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //sortIndexByHash();
    CountingSortCUDA();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    reindexParticles();  //sf 重新调整粒子顺序！！！！！！！！！！！！！！！需要注意温度等信息需要在这函数内加入
	//sf 特别是下一个时间不长仍会用到的属性 比如温度

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    findCellRange();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    gpu_model::calculateBlockRequirementSMSMode(d_block_reqs_, d_start_index_, d_end_index_, 32, numc_);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    prescanArrayRecursiveInt(d_breqs_offset_, d_block_reqs_, numc_, 0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    arrangeBlockTasks();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    buff_list_.swapObj(buff_temp_);

    //CUDA_SAFE_CALL(cudaEventRecord(end));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //float time;
    //CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, end));
    //std::cout << "gpu model time: " << time << "ms" << std::endl;
    //CUDA_SAFE_CALL(cudaEventDestroy(start));
    //CUDA_SAFE_CALL(cudaEventDestroy(end));

  //  return 0;
}

void Arrangement::countNum()
{

    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    //CUDA_SAFE_CALL(cudaMemset(cell_num_, 0x00000000, numc_ * sizeof(int)));
    countCellNum << <num_block, num_thread >> >(d_hash_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_, cell_num_, nump_);
    //countCellNum << <num_block, num_thread >> >(d_hash_, cell_num_, nump_, buff_list_.get_buff_list().position, cell_size_, grid_size_);
}


int xixi = 0;
void Arrangement::CountingSort_O()
{



    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numc_ + 1, num_thread);

    clean_data << <num_blockc, num_thread >> >(d_cell_nump_, numc_);

    CountingSort_Cell_Sum << <num_block, num_thread >> >(d_p_offset_, d_hash_, d_cell_nump_, nump_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_);

    cudaMemcpy(d_cell_offset_ + 1, d_cell_nump_, sizeof(int) * numc_, cudaMemcpyDeviceToDevice);
    CountingSort_Offest_P(num_blockc, num_thread, d_cell_offset_, numc_ + 1);








  //  xixi++;
   








    CountingSort_Result << <num_block, num_thread >> >(d_p_offset_p, d_p_offset_, d_hash_, hashp, d_cell_offset_, nump_, buff_list_.get_buff_list(), buff_temp_.get_buff_list());



	//std::ofstream out("xixixixixixixixixixixixixi.txt");


	//int *cellnum = new int[numc_];
	//int *celloff = new int[numc_];
	//cudaMemcpy(celloff, d_cell_offset_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(cellnum, d_cell_nump_, numc_*sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < numc_; i++){
	//	out << cellnum[i] << "   " << celloff[i] << std::endl;
	//}





	//copy_data << <num_block, num_thread >> >(d_hash_, d_index_, hashp, indexp, nump_);
	//cudaMemcpy(d_hash_, hashp, sizeof(int) * nump_, cudaMemcpyDeviceToDevice);




    //copy_data << <num_block, num_thread >> >(d_hash_, d_index_, hashp, indexp, nump_);
    //cudaMemcpy(d_hash_, hashp, sizeof(int) * nump_, cudaMemcpyDeviceToDevice);

    int *p = d_hash_;
    d_hash_ = hashp;
    hashp = p;
    buff_list_.swapObj(buff_temp_);









 /*   int nc = nump_;
    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *cof = new int[nc];
    int *cnp = new int[nc];

    int *cnmb = new int[numc_];

    int *brq = new int[numc_];
    cudaMemcpy(cof, d_p_offset_p, nc*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnmb, d_cell_nump_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnp, d_hash_, nc*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(brq, d_block_reqs_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nc; i++){
        out << cof[i] << "      " << cnp[i] << "     ";// << brq[cnp[i]] << std::endl;//;
        if (cnp[i] >= numc_){
            out << cnmb[cnp[i] - numc_] << "    " << brq[cnp[i] - numc_] << std::endl;
            //     out <<brq[cnp[i] - numc_] << std::endl;
        }
        else{
            out << cnmb[cnp[i]] << "    " << brq[cnp[i]] << std::endl;
            //         out << brq[cnp[i]] << std::endl;
        }
    }
    for (int i = 0; i < numc_; i++){
        //           out << brq[i] << std::endl;
    }
    std::cout << "---------------- ";*/














}

__global__ void CountingSort_Result_M(int *p_offset_p, int *p_offset, int *hash, int *hash_new, int *cell_offset, int num, ParticleBufferList old_data, ParticleBufferList new_data)
{
	int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (id < num)
	{
		int x_id = hash[id];
		int iStart = cell_offset[x_id];
		int p_id = iStart + p_offset[id];
		hash_new[p_id] = x_id;
		int ici = (x_id & 0xffffffc0);
		p_offset_p[p_id] = p_offset[id] + iStart - cell_offset[ici];

		new_data.position_d[p_id] = old_data.position_d[id];
		new_data.velocity[p_id] = old_data.velocity[id];
		new_data.evaluated_velocity[p_id] = old_data.evaluated_velocity[id];
		new_data.color[p_id] = old_data.color[id];
		//        new_data.phase[p_id] = old_data.phase[id];
	}
}
__global__ void CountingSort_Cell_SumM(int *p_offset, int *hashId, int *cell_numbers, int iSize, float4 *position, float cell_size, ushort3 grid_size)
{
	int x_id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;

	if (x_id < iSize)
	{
		int hid = ParticlePos2CellIdxM(position[x_id], grid_size, cell_size);
		hashId[x_id] = hid;// >> 6;
		p_offset[x_id] = atomicAdd(cell_numbers + hid, 1);
	}
}
__global__ void calculate_cell_info(int *cell_Numx, int *cell_Offset, int *cell_Offset_M, int numc)
{
	int id = __umul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (id < numc)
	{
		int idi = (id << 6);
		cell_Numx[id] = cell_Offset_M[idi + 64] - cell_Offset_M[idi];
		cell_Offset[id] = cell_Offset_M[idi];
	}
}
void Arrangement::CountingSort_O_M()
{

	int num_thread = 256;
	int num_block = ceil_int(nump_, num_thread);
	int numCN = (numc_ <<6);
	int num_blockc = ceil_int(numCN + 1, num_thread);

	cudaMemset(d_cell_nump_M, 0x00, sizeof(int)* (numCN+1));

	//clean_data << <num_blockc, num_thread >> >(d_cell_nump_M, numCN);
	
	CountingSort_Cell_SumM << <num_block, num_thread >> >(d_p_offset_, d_hash_, d_cell_nump_M, nump_, buff_list_.get_buff_list().position_d, cell_size_, grid_size_);

	//cudaMemcpy(d_cell_offset_M + 1, d_cell_nump_M, sizeof(int)* numCN, cudaMemcpyDeviceToDevice);
	//CountingSort_Offest_P(num_blockc, num_thread, d_cell_offset_M, numCN + 1);


	thrust::exclusive_scan(thrust::device_ptr<int>(d_cell_nump_M), thrust::device_ptr<int>(d_cell_nump_M) +numCN + 1, thrust::device_ptr<int>(d_cell_offset_M));

	CountingSort_Result_M << <num_block, num_thread >> >(d_p_offset_p, d_p_offset_, d_hash_, hashp, d_cell_offset_M, nump_, buff_list_.get_buff_list(), buff_temp_.get_buff_list());

	calculate_cell_info << <ceil_int(numc_, num_thread), num_thread >> >(d_cell_nump_, d_cell_offset_, d_cell_offset_M, numc_);

	//std::ofstream out("xixixixixixixixixixixixixi.txt");


	//int *cellnum = new int[numc_];
	//int *celloff = new int[numc_];
	//cudaMemcpy(celloff, d_cell_offset_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(cellnum, d_cell_nump_, numc_*sizeof(int), cudaMemcpyDeviceToHost);

	//for (int i = 0; i < numc_; i++){
	//	out << cellnum[i] << "   " << celloff[i] << std::endl;
	//}

	int *p = d_hash_;
	d_hash_ = hashp;
	hashp = p;
	buff_list_.swapObj(buff_temp_);
}


void Arrangement::CountingSortCUDA_Two9()
{
    int numCell = numc_ << 1;
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    int num_blockc = ceil_int(numCell, num_thread);
    clean_data << <num_blockc, num_thread >> >(cell_num_two, numCell);
    CountingSort_Cell_Sum_two << <num_block, num_thread >> >(d_p_offset_p, d_hash_, cell_num_two, nump_, d_block_reqs_, numc_);

 //   cudaDeviceSynchronize();

 //   xixi++;
/*    if (xixi == -1){
        int nc = nump_;
        std::ofstream out("xixixixixixixixixixixixixi.txt");
        int *cof = new int[nc];
        int *cnp = new int[nc];

        int *cnmb = new int[numc_];

        int *brq = new int[numc_];
        cudaMemcpy(cof, d_p_offset_p, nc*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cnmb, d_cell_nump_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cnp, d_hash_, nc*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(brq, d_block_reqs_, numc_*sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < nc; i++){
            out << cof[i] << "      " << cnp[i] << "     ";// << brq[cnp[i]] << std::endl;//;
            if (cnp[i] >= numc_){
                out << cnmb[cnp[i] - numc_]<<"    "<<brq[cnp[i] - numc_] << std::endl;
           //     out <<brq[cnp[i] - numc_] << std::endl;
            }
            else{
                out << cnmb[cnp[i]] << "    " << brq[cnp[i]] << std::endl;
       //         out << brq[cnp[i]] << std::endl;
            }
        }
        for (int i = 0; i < numc_; i++){
 //           out << brq[i] << std::endl;
        }
        std::cout << "---------------- ";
    }*/










    CountingSort_Offest_P(num_blockc, num_thread, cell_num_two, numCell);
    CountingSort_Result_two9 << <num_block, num_thread >> >(d_p_offset_p, d_hash_, hashp, cell_num_two, d_index_, nump_);
    unsigned int shared_mem_size = (num_thread + 1) * sizeof(int);
    knFindHybridModeMiddleValue << <num_block, num_thread, shared_mem_size >> >(numc_, d_middle_value_, hashp, nump_);
    CUDA_SAFE_CALL(cudaMemcpy(&middle_value_, d_middle_value_, sizeof(int), cudaMemcpyDeviceToHost));
}

void Arrangement::CountingSortCUDA_Two9_M()
{
	int numCell = numc_ << 1;
	int num_thread = 256;
	int num_block = ceil_int(nump_, num_thread);
	int num_blockc = ceil_int(numCell, num_thread);

	cudaMemset(cell_num_two, 0x00, sizeof(int)* (numCell));

	//clean_data << <num_blockc, num_thread >> >(cell_num_two, numCell+1);
	
	CountingSort_Cell_Sum_two_M << <num_block, num_thread >> >(d_hash_p, d_p_offset_p, d_hash_, cell_num_two, nump_, d_block_reqs_, numc_);
	CountingSort_Offest_P(num_blockc, num_thread, cell_num_two, numCell);

	//thrust::inclusive_scan(thrust::device_ptr<int>(cell_num_two), thrust::device_ptr<int>(cell_num_two) +numCell, thrust::device_ptr<int>(cell_num_two));


	CountingSort_Result_two9 << <num_block, num_thread >> >(d_p_offset_p, d_hash_p, hashp, cell_num_two, d_index_, nump_);
	unsigned int shared_mem_size = (num_thread + 1) * sizeof(int);
	knFindHybridModeMiddleValue << <num_block, num_thread, shared_mem_size >> >(numc_, d_middle_value_, hashp, nump_);
	CUDA_SAFE_CALL(cudaMemcpy(&middle_value_, d_middle_value_, sizeof(int), cudaMemcpyDeviceToHost));
}

int Arrangement::arrangeHybridMode9(){
    CountingSort_O();
    gpu_model::calculateBlockRequirementHybridMode(cell_type, d_cell_nump_, d_block_reqs_, p_gpu_model_,d_cell_offset_, d_cell_nump_,grid_size_, 32);
    CountingSortCUDA_Two9();
    prescanArrayRecursiveInt(d_task_array_offset_32_, d_block_reqs_, numc_, 0);
    arrangeBlockTasksFixed(d_block_task_, d_block_reqs_, d_task_array_offset_32_, 32);
    return (middle_value_ > nump_ || middle_value_ < 0) ? nump_ : middle_value_;
}
int Arrangement::arrangeHybridMode9M(){
	CountingSort_O_M();
	gpu_model::calculateBlockRequirementHybridMode(cell_type, d_cell_nump_, d_block_reqs_, p_gpu_model_, d_cell_offset_, d_cell_nump_, grid_size_, 32);
	CountingSortCUDA_Two9_M();
	//prescanArrayRecursiveInt(d_task_array_offset_32_, d_block_reqs_, numc_, 0);
	thrust::exclusive_scan(thrust::device_ptr<int>(d_block_reqs_), thrust::device_ptr<int>(d_block_reqs_) +numc_, thrust::device_ptr<int>(d_task_array_offset_32_));
	arrangeBlockTasksFixedM(d_hash_, d_cell_offset_, d_cell_nump_, d_block_task_, d_block_reqs_, d_task_array_offset_32_, 32);
	return (middle_value_ > nump_ || middle_value_ < 0) ? nump_ : middle_value_;
}
int Arrangement::arrangeHybridMode(){
    //calculateHash();
    //CountingSort_O();






    //CountingSortCUDA();
    countNum();
    //findCellRange();
    gpu_model::calculateBlockRequirementHybridMode(cell_type, cell_num_, d_block_reqs_, p_gpu_model_,
                                                   d_cell_offset_, d_cell_nump_,
                                                   grid_size_, 32);

    // sort particles
    //calculateHashWithBlockReq();
    //sortIndexByHash();
    CountingSortCUDA_Two();
    

//    std::cout << xixi << "----------OOOOOOOOOOOOOOOOOO" << std::endl;

    //reindexParticles();
    // get block tasks
    //findCellRangeAndHybridModeMiddleValue();
    //prescanArrayRecursiveInt(d_breqs_offset_, d_block_reqs_, numc_, 0);
    //arrangeBlockTasks();
    //CUDA_SAFE_CALL(cudaEventRecord(end));
    //CUDA_SAFE_CALL(cudaMemcpy(&middle_value_, d_middle_value_, sizeof(int), cudaMemcpyDeviceToHost));

    //buff_list_.swapObj(buff_temp_);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //float time;
    //CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, end));
    //std::cout << "gpu model time: " << time << "ms" << std::endl;
    //CUDA_SAFE_CALL(cudaEventDestroy(start));
    //CUDA_SAFE_CALL(cudaEventDestroy(end));

    prescanArrayRecursiveInt(d_task_array_offset_32_, d_block_reqs_, numc_, 0);




    /*unsigned int nc = numc_;
    std::ofstream out("xixixixixixixixixixixixixi.txt");
    int *ha = new int[nump_];
    int *cof = new int[nc];
    int *cnp = new int[nc];
    cudaMemcpy(ha, d_hash_, nump_*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cof, d_cell_nump_, nc*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cnp, d_cell_offset_, nc*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numc_; i++){
        for (int j = 0; j < d_cell_nump_[i]; j++){
                out << d_cell_offset_[j] << "  " << d_cell_nump_[j];
        }
        out<<std::endl;;
    }
    std::cout << middle_value_;
    std::cout << "---------------- ";*/
    






    arrangeBlockTasksFixed(d_block_task_, d_block_reqs_, d_task_array_offset_32_, 32);
    return (middle_value_ > nump_ || middle_value_ < 0) ? nump_ : middle_value_;
}


int* Arrangement::getDevCellStartIdx()
{
    return d_start_index_;
}

int* Arrangement::getDevCellEndIdx()
{
    return d_end_index_;
}

int Arrangement::getNumBlockSMSMode()
{
    return h_num_cta_;//h_num_block_;
}

BlockTask * Arrangement::getBlockTasks()
{
    return d_block_task_;
}

void Arrangement::resetNumParticle(unsigned int nump)
{
	nump_ = nump;

	if (nump_capacity_ < nump_)
	{
		CUDA_SAFE_CALL(cudaFree(d_hash_));
		CUDA_SAFE_CALL(cudaFree(d_index_));
		CUDA_SAFE_CALL(cudaMalloc(&d_hash_, nump_ * sizeof(int)));
		CUDA_SAFE_CALL(cudaMalloc(&d_index_, nump_ * sizeof(int)));
		nump_capacity_ = nump_;
	}
}

void Arrangement::calculateHash()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);

    knCalculateHash<<<num_block, num_thread>>>(d_hash_, d_index_, 
                                               buff_list_.get_buff_list().position_d, 
                                               cell_size_, grid_size_, nump_);
}

void Arrangement::calculateHashWithBlockReq()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);
    knCalculateHashWithBlockReq << <num_block, num_thread >> >(numc_, d_hash_, d_index_, d_block_reqs_, nump_);
}

void Arrangement::sortHash()
{
    if (0 == nump_) return;

    thrust::sort(thrust::device_ptr<int>(d_hash_),
                 thrust::device_ptr<int>(d_hash_ + nump_));
}

void Arrangement::sortIndexByHash()
{
    if (0 == nump_) return;

	const int size = nump_;
	int *ha = new int[nump_];
	int *in = new int[nump_];
	cudaMemcpy(ha, d_hash_, size * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(in, d_index_, size * sizeof(int), cudaMemcpyDeviceToHost);
	thrust::sort_by_key(ha, ha + nump_, in);
	cudaMemcpy(d_hash_, ha, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_, in, size * sizeof(int), cudaMemcpyHostToDevice);

	free(ha);
	free(in);

    /*thrust::sort_by_key(thrust::device_ptr<int>(d_hash_),
                        thrust::device_ptr<int>(d_hash_ + nump_),
                        thrust::device_ptr<int>(d_index_));*/
}

void Arrangement::reindexParticles()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);

    knReindexParticles<<<num_block, num_thread>>>(buff_list_.get_buff_list(), buff_temp_.get_buff_list(), d_index_, nump_);
}

void Arrangement::reindexParticles2()
{

}

void Arrangement::findCellRange()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);

    CUDA_SAFE_CALL(cudaMemset(d_start_index_, 0xffffffff, numc_ * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_end_index_, 0x0, numc_ * sizeof(int)));

    unsigned int shared_mem_size = (num_thread + 1) * sizeof(int);

    knFindCellRange<<<num_block, num_thread, shared_mem_size>>>(d_start_index_, 
                                                                d_end_index_, 
                                                                d_hash_, nump_);
}

void Arrangement::findCellRangeAndHybridModeMiddleValue()
{
    int num_thread = 256;
    int num_block = ceil_int(nump_, num_thread);

    CUDA_SAFE_CALL(cudaMemset(d_start_index_, 0xffffffff, numc_ * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_end_index_, 0x0, numc_ * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_middle_value_, 0xffffffff, sizeof(int)));

    unsigned int shared_mem_size = (num_thread + 1) * sizeof(int);

    knFindCellRangeAndHybridModeMiddleValue << <num_block, num_thread, shared_mem_size >> >
        (numc_, d_start_index_, d_end_index_, d_middle_value_, d_hash_, nump_);
}

void Arrangement::insertParticles()
{

}

void Arrangement::arrangeBlockTasks()
{
    int num_thread = 256;
    int num_block = ceil_int(numc_, num_thread);

//    knArrangeBlockTasks<<<num_block, num_thread>>>(d_block_task_, d_num_block_, d_block_reqs_, d_breqs_offset_, grid_size_, numc_);

    CUDA_SAFE_CALL(cudaMemcpy(&h_num_block_, d_num_block_, sizeof(int), cudaMemcpyDeviceToHost));
}

}
//
// sph_arrangement.cuh
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _SPH_ARRANGEMENT_H
#define _SPH_ARRANGEMENT_H

#include "particle/particle_buffer.h"

namespace sph
{

struct SmsTaskPairStats
{
    int total_pairs = 0;
    int same_cell_pairs = 0;
    int is_same_pairs = 0;
    int full_is_same_pairs = 0;
    int partial_is_same_pairs = 0;
    int compact_is_same_pairs = 0;
    int widened_is_same_pairs = 0;
};

class Arrangement
{
public:
    Arrangement(ParticleBufferObject &buff_list,
                ParticleBufferObject &buff_temp,
                unsigned int nump,
                unsigned int nump_capacity,
                float inv_cell_size,
                ushort3 grid_size);

    ~Arrangement();

    // Live per-frame arrangement path.
    void arrangeHybridFrame();
    int getSmsTaskCount() const;
    SmsTaskPairStats getSmsTaskPairStats() const;
    const BlockTask *getSmsTasks() const;
    SameCellForceAccum *getSameCellForceAccum() const {
        return d_same_cell_force_accum_;
    }

    int *getDevCellOffset() { return d_cell_offset_; }
    int *getDevCellOffsetM() { return d_cell_offset_M; }
    int *getDevCellIndex() { return d_index_; }
    int *getDevCellNumP() { return d_cell_nump_; }

    const int *getDeviceSmsTaskCount() const { return d_num_cta_; }
    const int *getDeviceTraParticleCount() const { return d_middle_value_; }
    int getTraParticleCount() const { return middle_value_; }
    unsigned int getNumC() const { return numc_; }

    void resetNumParticle(unsigned int nump);

    // Legacy arrangement entry points retained for inactive solver variants.
    int arrangeTRAMode();
    void arrangeSMSMode();
    int arrangeHybridMode();
    int arrangeHybridMode9();
    void test();

    void sortParticles();
    void assignTasksFixedCTA();

    int *getDevCellStartIdx();
    int *getDevCellEndIdx();
    int *getDevOffsetData() { return d_cell_offset_data; }

    void CountingSortCUDA();
    void CountingSort_O();

    void CountingSortCUDA_Two();
    void CountingSortCUDA_Two9();
    void countNum();

    void CountingSort_O_M();
    void CountingSortCUDA_Two9_M();

private:
    void calculateHash();
    void calculateHashWithBlockReq();
    void sortHash();
    void sortIndexByHash();
    void reindexParticles();    // use index sorted by hash to reindex
    void reindexParticles2();   // use "particle offset in cell" and "prefix summed cell offset" to reindex
    void findCellRange();
    void findCellRangeAndHybridModeMiddleValue();
    void insertParticles();
    void arrangeBlockTasks();

    void CSInsertParticles();
    void CSCountingSortFull();
    void arrangeSmsTasks(
        const int *hash, const int *cell_offsets,
        const int *cell_particle_counts, BlockTask *tasks,
        const int *cell_task_counts, const int *cell_task_offsets);
    void arrangeBlockTasksFixed(BlockTask* d_task_array, int* d_cta_reqs,
                                int* d_task_array_offset, int cta_size);
    void arrangeBlockTasksFloat();
    void allocateCubTempStorage();

    void CSCalculateRequiredCTAsFixed(int *cat_offset, int* d_cta_reqs, int cta_size);

    ParticleBufferObject &buff_list_; // particle device buffer
    ParticleBufferObject &buff_temp_; // particle device buffer for replacement 
    unsigned int nump_;             // #particles
	unsigned int nump_capacity_;
    unsigned int numc_;             // #cells
    float cell_size_;
    float inv_cell_size_;           // 1.0f / cell_size_, precomputed to avoid per-thread division
    ushort3 grid_size_;
    int middle_value_ = 0;
    int* h_middle_value_pinned_ = nullptr; // [1] pinned host buffer for async D2H of middle_value_
    int* h_hybrid_counts_pinned_ = nullptr; // [2] contiguous middle/task-count readback

    int  h_num_cta_ = 0;
    int* h_num_cta_pinned_ = nullptr;      // [1] pinned host buffer for async D2H of d_num_cta_

    int* d_num_cta_;
    int* d_hybrid_counts_ = nullptr; // [2]: middle value, SMS task count
    int* d_cell_offset_;            // [numc] the offset in memory of the particles in each cell
    int* d_cell_nump_;              // [numc] the number of particles in each cell
    int* d_p_offset_;

    int* d_p_offset_p;

    int* d_cell_offset_data;

    int *d_start_index_;            // [numc]device buffer, cell start index
    int *d_end_index_;              // [numc]device buffer, cell end index
    int *d_hash_;                   // [nump]
    int *d_index_;                  // [nump]
    int *d_index_alt_;              // persistent CUB SortPairs value output
    
    int *hashp;                   // [nump]
	int *d_hash_p;                   // [nump]
   // int *indexp;
    int *cell_num_;
    int *cell_num_two;
    int *cell_type;
	int* d_cell_nump_M;
	int* d_cell_offset_M;


    int* d_task_array_offset_32_;   // [numc]result of prescan
    int *d_block_reqs_;             // [numc]for SMS Mode
    int *d_breqs_offset_;           // [numc]result of prescan
    int *d_num_block_;              // [1]
    int h_num_block_ = 0;
    BlockTask *d_block_task_;       // [numb]
    SameCellForceAccum *d_same_cell_force_accum_ = nullptr;
    
    int *d_middle_value_;           // [1]for Hybrid Mode


    // One persistent device allocation shared by CUB scans and radix sorts.
    void *d_cub_temp_ = nullptr;
    size_t cub_temp_bytes_ = 0;

};

}

#endif/*_SPH_ARRANGEMENT_CUH*/

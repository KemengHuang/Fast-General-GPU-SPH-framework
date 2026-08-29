//
// sph_utils.cuh
// Hybrid_Parallel_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _SPH_UTILS_CUH
#define _SPH_UTILS_CUH

#include <math.h>
#include <cuda_runtime.h>

// When 1, the hybrid density/force kernels read the TRA/SMS split point and the
// SMS task count from device memory and are launched with a conservatively
// over-provisioned grid (excess blocks exit immediately), so a simulation frame
// needs no host-side synchronization. When 0 (default), the host reads the two
// scalars back once per frame with a single stream sync and launches exact grids.
// Measured on RTX 4090 with 3.94M particles, the exact-grid path is slightly
// faster (~2%): scheduling the ~200k over-provisioned blocks costs more than
// the single mid-frame sync does.
#ifndef HYBRID_DEVICE_GRID_SIZING
#define HYBRID_DEVICE_GRID_SIZING 0
#endif

// The live SMS kernels treat the two warps independently (isSame is fixed to
// zero), so keep each 32-particle task's original micro-cell bounds.  The old
// pairing logic widened the first task to cover both warps and increased the
// number of rejected neighbor pairs.
#ifndef SMS_PAIR_TASK_COALESCING
#define SMS_PAIR_TASK_COALESCING 0
#endif

// Number of self particles handled by one register-path cooperative group.
// Candidate builds may set this to 16 or 8; the production/shared baseline is
// 32.  A 64-thread block always packs an integral number of groups.
#ifndef SMS_TASK_PARTICLES
#define SMS_TASK_PARTICLES 32
#endif

#ifndef SMS_BLOCK_THREADS
#define SMS_BLOCK_THREADS 64
#endif

#if SMS_TASK_PARTICLES != 8 && SMS_TASK_PARTICLES != 16 && SMS_TASK_PARTICLES != 32
#error "SMS_TASK_PARTICLES must be 8, 16, or 32"
#endif

#if SMS_BLOCK_THREADS != 64 && SMS_BLOCK_THREADS != 128 && SMS_BLOCK_THREADS != 256
#error "SMS_BLOCK_THREADS must be 64, 128, or 256"
#endif

#define SMS_TASKS_PER_BLOCK (SMS_BLOCK_THREADS / SMS_TASK_PARTICLES)
#define SMS_MIN_BLOCKS_PER_SM (640 / SMS_BLOCK_THREADS)

namespace sph
{

	const int kInvalidCellIdx = 0xffffffff;

	__device__ __host__
		inline int ceil_int(int a, int b) { return (a + b - 1) / b; }

	__device__
		inline ushort3 ParticlePos2CellPos(const float4 &pos, float inv_cell_size)
	{
			return make_ushort3(floorf(pos.x * inv_cell_size),
				floorf(pos.y * inv_cell_size),
				floorf(pos.z * inv_cell_size));
		}
	__device__
		inline ushort3 ParticlePos2CellPosM(const float4 &pos, float inv_cell_size)
	{
			float rat = 4.f * inv_cell_size;
			return make_ushort3(floorf(pos.x *rat),
				floorf(pos.y *rat),
				floorf(pos.z *rat));
		}
	__device__
		inline int CellPos2CellIdx(const ushort3 &cell_pos, const ushort3 &grid_size)
	{
			if (cell_pos.x >= grid_size.x || cell_pos.x < 0 ||
				cell_pos.y >= grid_size.y || cell_pos.y < 0 ||
				cell_pos.z >= grid_size.z || cell_pos.z < 0)
				return kInvalidCellIdx;
			return cell_pos.x + grid_size.x * (cell_pos.y + grid_size.y * cell_pos.z);
		}

	__device__
		inline int ParticlePos2CellIdx(const float4 &pos, const ushort3 &grid_size, float inv_cell_size)
	{
			ushort3 cell_pos = ParticlePos2CellPos(pos, inv_cell_size);
			return CellPos2CellIdx(cell_pos, grid_size);
		}
	__device__
		inline int CellPos2CellIdxM(const ushort3 &cell_pos, const ushort3 &grid_size)
	{
			if (cell_pos.x >= (grid_size.x << 2) || cell_pos.x < 0 ||
				cell_pos.y >= (grid_size.y << 2) || cell_pos.y < 0 ||
				cell_pos.z >= (grid_size.z << 2) || cell_pos.z < 0)
				return kInvalidCellIdx;

			int x = cell_pos.x & 0x03;
			int y = cell_pos.y & 0x03;
			int z = cell_pos.z & 0x03;
			int xx = (cell_pos.x >> 2);
			int yy = (cell_pos.y >> 2);
			int zz = (cell_pos.z >> 2);

			int idc = xx + grid_size.x * (yy + grid_size.y * zz);

		//	int idi = (x << 4) | (y & 0x01) | ((y & 0x02) << 1) | ((z & 0x01) << 1) | ((z & 0x02) << 2);

			int idi = y + ((z + (x<<2))<<2);
			int id = (idc << 6) | (idi);
			return id;
		}
	__device__
		inline int ParticlePos2CellIdxM(const float4 &pos, const ushort3 &grid_size, float inv_cell_size)
	{
			ushort3 cell_pos = ParticlePos2CellPosM(pos, inv_cell_size);
			return CellPos2CellIdxM(cell_pos, grid_size);
		}
	__device__
		inline ushort3 CellIdx2CellPos(int idx, const ushort3 &grid_size)
	{
			return make_ushort3(idx % grid_size.x,
				idx / grid_size.x % grid_size.y,
				idx / grid_size.x / grid_size.y);
		}

}

#endif/*_SPH_UTILS_CUH*/

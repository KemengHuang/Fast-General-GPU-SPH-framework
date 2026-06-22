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

namespace sph
{

	const int kInvalidCellIdx = 0xffffffff;

	__device__ __host__
		inline int ceil_int(int a, int b) { return (a + b - 1) / b; }

	__device__
		inline ushort3 ParticlePos2CellPos(const float4 &pos, float cell_size)
	{
			return make_ushort3(floorf(pos.x / cell_size),
				floorf(pos.y / cell_size),
				floorf(pos.z / cell_size));
		}
	__device__
		inline ushort3 ParticlePos2CellPosM(const float4 &pos, float cell_size)
	{
			float rat = 4.f / cell_size;
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
		inline int ParticlePos2CellIdx(const float4 &pos, const ushort3 &grid_size, float cell_size)
	{
			ushort3 cell_pos = ParticlePos2CellPos(pos, cell_size);
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
		inline int ParticlePos2CellIdxM(const float4 &pos, const ushort3 &grid_size, float cell_size)
	{
			ushort3 cell_pos = ParticlePos2CellPosM(pos, cell_size);
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

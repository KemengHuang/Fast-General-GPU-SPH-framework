//
// cuda_math.cuh
// Heterogeneous_SPH
//
// created by kmhuang and ruanjm on 2018/09/01
// Copyright (c) 2019 kmhuang and ruanjm. All rights reserved.
//

#ifndef _CUDA_MATH_H
#define _CUDA_MATH_H

#include <cuda_runtime.h>

#define  kFloatSmall    (1e-12f)

__host__ __device__ 
inline uint3 operator+(const uint3 &a, const uint3 &b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline uint3 operator+(const uint3 &a, const int3 &b)
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline int3 operator+(const int3 &a, const int3 &b)
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline ushort3 operator+(const ushort3 &a, const ushort3 &b)
{
	return make_ushort3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__
inline float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__
inline float3 operator/(const float3 &a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__host__ __device__
inline void operator-=(float3 &a, const float3 &b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

__host__ __device__
inline void operator+=(float3 &a, const float3 &b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

__host__ __device__
inline void operator*=(float3 &a, const float b)
{
	a.x *= b; a.y *= b; a.z *= b;
}

__host__ __device__
inline float distance_square(const float4 &a, const float4 &b)
{
    float deltax = a.x - b.x;
    float deltay = a.y - b.y;
    float deltaz = a.z - b.z;
    return deltax* deltax + deltay * deltay + deltaz * deltaz;
}

#endif/*_CUDA_MATH_H*/
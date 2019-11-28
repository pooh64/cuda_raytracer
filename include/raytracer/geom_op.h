#pragma once

#include <cuda_runtime.h>

__host__ __device__
inline float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__
inline float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__
inline float3 mul(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__host__ __device__
inline float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y,
			   a.z * b.x - a.x * b.z,
			   a.x * b.y - a.y * b.x);
}

__host__ __device__
inline float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}

__host__ __device__
inline float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline float3 normalize(float3 v)
{
	float inv_len = rsqrtf(dot(v, v));
	return mul(inv_len, v);
}

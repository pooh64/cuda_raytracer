#pragma once

#include <iostream>
#include <cuda_runtime.h>

__host__ __device__ inline
float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline
float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline
float3 operator*(float3 a, float3 b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline
float3 operator*(float val, float3 v)
{
	return make_float3(val * v.x, val * v.y, val * v.z);
}

__host__ __device__ inline
float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline
float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - a.z * b.y,
			   a.z * b.x - a.x * b.z,
			   a.x * b.y - a.y * b.x);
}

__host__ __device__ inline
float3 normalize(float3 v)
{
	float inv_len = rsqrtf(dot(v, v));
	return inv_len * v;
}

__host__ __device__ inline
float3 clamp(float3 v)
{
	v.x = (v.x > 1.0f) ? 1.0f : v.x;
	v.y = (v.y > 1.0f) ? 1.0f : v.y;
	v.z = (v.z > 1.0f) ? 1.0f : v.z;
	return v;
}

inline std::ostream &operator<<(std::ostream &os, const float3 &v)
{
	os << "(" << v.x << " " << v.y << " " << v.z << ")";
	return os;
}

/* float2 */

__host__ __device__ inline
float2 operator+(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__host__ __device__ inline
float2 operator-(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

__host__ __device__ inline
float2 operator*(float2 a, float2 b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

__host__ __device__ inline
float2 operator*(float val, float2 v)
{
	return make_float2(val * v.x, val * v.y);
}

__host__ __device__ inline
float dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}

__host__ __device__ inline
float2 normalize(float2 v)
{
	float inv_len = rsqrtf(dot(v, v));
	return inv_len * v;
}

__host__ __device__ inline
float2 clamp(float2 v)
{
	v.x = (v.x > 1.0f) ? 1.0f : v.x;
	v.y = (v.y > 1.0f) ? 1.0f : v.y;
	return v;
}

inline std::ostream &operator<<(std::ostream &os, const float2 &v)
{
	os << "(" << v.x << " " << v.y << ")";
	return os;
}

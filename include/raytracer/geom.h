#pragma once

#include <raytracer/floatvec.h>

struct Ray {
	float3 o, d;
	__host__ __device__
	float3 point_at(float t) const
	{
		return o + t * d;
	}
};

struct HitRecord {
	float  t;
	float3 n;
};

struct Shape {
	__host__ __device__
	virtual bool        hit(const Ray &ray, float tmin, float tmax,
		HitRecord *rec) const =0;
	__host__ __device__
	virtual bool shadow_hit(const Ray &ray, float tmin, float tmax)
				const =0;
};

struct Sphere {
	float3 o;
	float  r;

#define __SPHERE_HIT_CODE				\
		float3 tmp = ray.o - o;			\
		double a = dot(ray.d, ray.d);		\
		double b = 2 * dot(ray.d, tmp);		\
		double c = dot(tmp, tmp) - r * r;	\
		double discr = b * b - 4 * a * c;	\
		if (discr < 0)				\
			return false;			\
		discr = sqrt(discr);			\
		double  t = (-b - discr) / (2 * a);	\
		if (t < tmin)				\
			t = (-b + discr) / (2 * a);	\
		if (t < tmin || t > tmax)		\
			return false;

	__host__ __device__
	bool hit(const Ray &ray, float tmin, float tmax,
		HitRecord *rec) const
	{
		__SPHERE_HIT_CODE
		rec->t = t;
		rec->n = normalize(ray.point_at(t) - o);
		return true;
	}

	__host__ __device__
	bool shadow_hit(const Ray &ray, float tmin, float tmax) const
	{
		__SPHERE_HIT_CODE
		return true;
	}
#undef __SPHERE_HIT_CODE
};

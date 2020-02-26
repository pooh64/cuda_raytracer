#pragma once

#include <raytracer/floatvec.h>

struct Ray {
	float3 o, d;
	float3 point_at(float t) const
	{
		return o + t * d;
	}
};

struct HitRecord {
	float  t;
	float3 n;
	float3 color;
};

struct Shape {
	virtual bool        hit(const Ray &ray, float tmin, float tmax,
		HitRecord &rec) const =0;
	virtual bool shadow_hit(const Ray &ray, float tmin, float tmax)
				const =0;
};

struct Sphere final : public Shape {
	float3 o;
	float  r;
	float3 color;

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

	bool hit(const Ray &ray, float tmin, float tmax,
		HitRecord &rec) const override
	{
		__SPHERE_HIT_CODE
		rec.t = t;
		rec.n = normalize(ray.point_at(t) - o);
		rec.color = color;
		return true;
	}

	bool shadow_hit(const Ray &ray, float tmin, float tmax) const override
	{
		__SPHERE_HIT_CODE
		return true;
	}
#undef __SPHERE_HIT_CODE
};

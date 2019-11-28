#pragma once

#include <raytracer/geom_op.h>

struct Ray {
	float3 pos;
	float3 dir;
};

struct Camera {
	float3 pos, dir, axis_x, axis_y;
	float ratio, fov;

	void set_screen(float fov_, float ratio_)
	{
		fov = fov_;
		ratio = ratio_;
	}

	void look_at(float3 const &eye, float3 const &at, float3 const &up)
	{
		pos    = eye;
		dir    = normalize(at - eye);
		axis_x = normalize(cross(dir, up));
		axis_y = cross(axis_x, dir);

		axis_x = mul(ratio * fov, axis_x);
		axis_y = mul(        fov, axis_y);
	}

	Ray cast(float x, float y)
	{
		return Ray { .pos = pos,
			     .dir = mul(x, axis_x) + mul(y, axis_y) + dir };
	}
};

struct Sphere {
	float3 pos;
	float  r2;
};

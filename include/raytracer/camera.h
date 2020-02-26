#pragma once

#include <raytracer/geom.h>

/*
 * fov - axis_y in rad
 * ratio = res_x/res_y */
struct Camera {
	float3 pos, dir, axis_x, axis_y;
	float ratio, fov;

	__host__ __device__
	void set_screen(float fov_, float ratio_)
	{
		fov = fov_;
		ratio = ratio_;
	}

	__host__ __device__
	void look_at(float3 const &eye, float3 const &at, float3 const &up)
	{
		pos    = eye;
		dir    = normalize(at - eye);
		axis_x = normalize(cross(dir, up));
		axis_y = cross(axis_x, dir);

		axis_x = (ratio * fov) * axis_x;
		axis_y =          fov  * axis_y;
	}

	__host__ __device__
	Ray cast(float x, float y)
	{
		return Ray { .o = pos,
			     .d = x * axis_x + y * axis_y + dir };
	}
};

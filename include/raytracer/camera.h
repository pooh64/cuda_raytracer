#pragma once

#include <raytracer/geom.h>

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

		axis_x = (ratio * fov) * axis_x;
		axis_y =          fov  * axis_y;
	}

	Ray cast(float x, float y)
	{
		return Ray { .pos = pos,
			     .dir = x * axis_x + y * axis_y + dir };
	}
};

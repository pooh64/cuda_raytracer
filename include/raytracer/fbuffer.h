#pragma once

#include <cstdint>
#include <raytracer/floatvec.h>

extern "C" {
#include <linux/fb.h>
}

struct Fbuffer : public fb_var_screeninfo, public fb_fix_screeninfo {
	union Color {
		struct { uint8_t b, g, r, a; };
		uchar4 val;
	};

	Color *buf;

	Color *operator[](std::uint32_t);
	Color const *operator[](std::uint32_t) const;

	int init(const char *path);
	int destroy();
	int update();

	void fill(Color c);
	void clear();

    private:
	int fd;
};

__host__ __device__
inline uchar4 to_uchar4color(float3 c3)
{
	c3 = 255 * c3;
	Fbuffer::Color rc;
	rc.b = c3.x;
	rc.g = c3.y;
	rc.r = c3.z;
	rc.a = 0;
	return rc.val;
}

__host__ __device__
inline float3 to_float3color(uchar4 val)
{
	Fbuffer::Color c;
	c.val = val;
	return (1.0f/255) * float3 { float(c.b), float(c.g), float(c.r) };
}

inline Fbuffer::Color *Fbuffer::operator[](std::uint32_t y)
{
	return buf + y * xres;
}

inline Fbuffer::Color const *Fbuffer::operator[](std::uint32_t y) const
{
	return buf + y * xres;
}

inline void Fbuffer::fill(Fbuffer::Color c)
{
	for (std::size_t i = 0; i < xres * yres; ++i)
		buf[i] = c;
}

inline void Fbuffer::clear()
{
	Color empty{ 0, 0, 0, 0 };
	for (std::size_t i = 0; i < xres * yres; ++i)
		buf[i] = empty;
}

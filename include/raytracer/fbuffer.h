#pragma once

#include <cstdint>

extern "C" {
#include <linux/fb.h>
}

struct Fbuffer : public fb_var_screeninfo, public fb_fix_screeninfo {
	struct Color {
		std::uint8_t b, g, r, a;
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

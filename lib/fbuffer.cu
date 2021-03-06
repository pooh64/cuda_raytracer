extern "C" {
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
};

#include <cstdlib>

#include <raytracer/fbuffer.h>

int Fbuffer::init(const char *path)
{
	fd = open(path, O_RDWR);
	if (fd < 0)
		goto handle_err_0;

	if (ioctl(fd, FBIOGET_VSCREENINFO, (fb_var_screeninfo *)this) < 0)
		goto handle_err_1;

	if (ioctl(fd, FBIOGET_FSCREENINFO, (fb_fix_screeninfo *)this) < 0)
		goto handle_err_1;

	buf = (Color*) malloc(sizeof(Color) * xres * yres);
	if (buf == NULL)
		goto handle_err_1;

	return 0;

handle_err_1:
	close(fd);
handle_err_0:
	return -1;
}

int Fbuffer::destroy()
{
	free(buf);
	return close(fd);
}

int Fbuffer::update()
{
	int rc = pwrite(fd, buf, xres * yres * sizeof(Color), 0);
	return rc < 0 ? rc : 0;
}

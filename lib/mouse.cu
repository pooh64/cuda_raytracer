extern "C" {
#include <fcntl.h>
#include <unistd.h>
}

#include <raytracer/mouse.h>

int Mouse::init(const char *path) noexcept
{
	return fd = open(path, O_RDONLY | O_NONBLOCK);
}

int Mouse::destroy() noexcept
{
	return close(fd);
}

bool Mouse::poll_ev(Mouse::Event &e) noexcept
{
	return read(fd, &e, sizeof(Mouse::Event)) > 0;
}

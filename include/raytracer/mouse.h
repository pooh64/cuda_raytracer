#pragma once

#include <cstdint>

struct Mouse {
	struct Event {
		uint8_t flags;
		int8_t dx, dy;

		bool LeftButton() const noexcept
		{
			return flags & 0x1;
		}
		bool RightButton() const noexcept
		{
			return flags & 0x2;
		}
		bool MiddleButton() const noexcept
		{
			return flags & 0x4;
		}
	};

	int init(const char *path) noexcept;
	int destroy() noexcept;

	bool poll_ev(Event &e) noexcept;

private:
	int fd;
};

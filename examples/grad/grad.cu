#include <cuda_runtime.h>
#include <cuda.h>
#include <raytracer/geom.h>
#include <raytracer/fbuffer.h>

#include <iostream>

struct render_param {
	float2 cvec[3];
};

__global__
void render(uchar4 *cbuf, uint cbuf_w, uint cbuf_h, render_param param)
{
	auto const &cvec = param.cvec;

	uint offs = blockDim.x * blockIdx.x + threadIdx.x;
	float2 pix { float(offs % cbuf_w) / cbuf_w - 0.5f,
		     float(offs / cbuf_w) / cbuf_h - 0.5f};

	uchar4 color = make_uchar4(((dot(pix, cvec[0])) + 0.5f) * 255,
				   ((dot(pix, cvec[1])) + 0.5f) * 255,
				   ((dot(pix, cvec[2])) + 0.5f) * 255, 255);
	cbuf[offs] = color;
}

int main()
{
	Fbuffer fb;
	if (fb.init("/dev/fb0") < 0) {
		std::cerr << "Can't open /dev/fb0\n";
		return 1;
	}

	uchar4 *device_cbuf;
	size_t cbuf_s = fb.xres * fb.yres;
	cudaMalloc(&device_cbuf, cbuf_s * sizeof(uchar4));

	uint block_dim = 256;
	uint grid_dim = cbuf_s / block_dim;

	for (float a = 0; ; a += 0.01f) {
		render_param param;
		for (int n = 0; n < 3; ++n)
			param.cvec[n] = make_float2(std::sin(a / (1 + n)) / sqrt(2),
						    std::cos(a / (1 + n)) / sqrt(2));

		render<<<grid_dim, block_dim>>>(device_cbuf, fb.xres, fb.yres, param);
		cudaDeviceSynchronize();
		cudaMemcpy(fb.buf, device_cbuf, cbuf_s * sizeof(uchar4), cudaMemcpyDeviceToHost);
		fb.update();
	}
	//cudaFree(device_cbuf);
	//return 0;
}

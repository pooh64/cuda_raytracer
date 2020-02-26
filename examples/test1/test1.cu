#include <cuda_runtime.h>
#include <cuda.h>
#include <raytracer/camera.h>
#include <raytracer/geom.h>
#include <raytracer/fbuffer.h>

#include <iostream>
#include <vector>

Sphere defined_spheres[] = {
	{.o = make_float3(0, 0, -1),
	 .r = 0.1,
	 .color = make_float3(0.8, 0.5, 0.3)},
};

struct Scene {
	Sphere *sph;
	size_t  n_sph;
};

__global__
void render(uchar4 *cbuf, uint cbuf_w, uint cbuf_h, Scene const *scene)
{
	uint offs = blockDim.x * blockIdx.x + threadIdx.x;
	float2 pix { float(offs % cbuf_w) / cbuf_w - 0.5f,
		     float(offs / cbuf_w) / cbuf_h - 0.5f};
	pix = 2 * pix;

	Camera cam;
	cam.set_screen(3.1415 / 3, float(cbuf_w) / cbuf_h);
	cam.look_at(make_float3(0, 0, 0),
		    make_float3(0, 0, -1),
		    make_float3(0, 1, 0));

	float tmax = FLT_MAX;
	float3 pix_color = make_float3(0.5, 0.5, 0.5);

	Ray ray = cam.cast(pix.x, pix.y);
	HitRecord hit;
	bool is_a_hit;
	for (uint i = 0; i < scene->n_sph; ++i) {
		if (scene->sph[i].hit(ray, 0, tmax, &hit)) {
			is_a_hit = true;
			tmax = hit.t;
		}
	}
	if (is_a_hit)
		pix_color = hit.color;

	cbuf[offs] = to_uchar4color(pix_color);
}

int main()
{
	Fbuffer fb;
	if (fb.init("/dev/fb0") < 0) {
		std::cerr << "Can't open /dev/fb0\n";
		return 1;
	}

	uchar4 *device_cbuf;
	size_t const cbuf_s = fb.xres * fb.yres;
	cudaMalloc(&device_cbuf, cbuf_s * sizeof(uchar4));

	Scene scene_data, *scene_device;
	scene_data.n_sph = sizeof(defined_spheres) / sizeof(*defined_spheres);
	size_t sph_sz = sizeof(*scene_data.sph) * scene_data.n_sph;
	cudaMalloc(&scene_data.sph, sph_sz);
	cudaMemcpy(scene_data.sph, defined_spheres, sph_sz, cudaMemcpyHostToDevice);
	cudaMalloc(&scene_device, sizeof(scene_data));
	scene_data.n_sph = sizeof(defined_spheres) / sizeof(*defined_spheres); // again
	cudaMemcpy(scene_device, &scene_data, sizeof(scene_data), cudaMemcpyHostToDevice);

	uint block_dim = 256;
	uint grid_dim = cbuf_s / block_dim;

	while (1) {
		render<<<grid_dim, block_dim>>>(device_cbuf, fb.xres, fb.yres, scene_device);
		cudaDeviceSynchronize();
		cudaMemcpy(fb.buf, device_cbuf, cbuf_s * sizeof(uchar4), cudaMemcpyDeviceToHost);
		fb.update();
	}
	//cudaFree(device_cbuf);
	//return 0;
}

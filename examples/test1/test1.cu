#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <raytracer/camera.h>
#include <raytracer/geom.h>
#include <raytracer/fbuffer.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

Sphere defined_spheres[] = {
	{.o = { 0, 0, -1 },
	 .r = 0.4,
	 .color = { 0.9, 0.6, 0.4 }},
	{.o = { 0.45, 0, -0.8 },
	 .r = 0.3,
	 .color = { 0.5, 0.9, 1.0 }},
	{.o = { 0.6, 0, -0.6 },
	 .r = 0.25,
	 .color = { 0.6, 1.0, 0.6 }},
	{.o = { -1.5, -0.7, -1.7 },
	 .r = 1.2,
	 .color = { 0.4, 0.5, 1.0 }},
	{.o = { 0, -0.6, -0.7 },
	 .r = 0.08,
	 .color = { 0.95, 0.95, 0.95 }},
};

struct Scene {
	Sphere *sph;
	uint  n_sph;
};

__device__
inline bool spheres_hit(Sphere const *arr, uint arr_sz, Ray const *ray, HitRecord *hit, uint *sph_id)
{
	bool is_a_hit = false;
	float tmax = FLT_MAX;
	for (uint i = 0; i < arr_sz; ++i) {
		if (i != *sph_id && arr[i].hit(*ray, 0, tmax, hit)) {
			is_a_hit = true;
			tmax = hit->t;
			*sph_id = i;
		}
	}
	return is_a_hit;
}

__global__
void setup_rnd_kernel(curandState_t *rnd_buf)
{
	uint const idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (blockIdx.x != 0)
		return;
	curand_init(1234, idx, 0, &rnd_buf[idx]);
}

__global__
void render(uchar4 *cbuf, uint cbuf_w, uint cbuf_h, Scene const *scene,
	curandState_t *rnd_buf)
{
	uint const idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t *my_rnd = &rnd_buf[idx % blockDim.x];
	float2 const pix = 2.0f * float2{ float(idx % cbuf_w) / cbuf_w - 0.5f,
					  float(idx / cbuf_w) / cbuf_h - 0.5f};

	Camera cam;
	cam.set_screen(3.1415 / 8, float(cbuf_w) / cbuf_h);
	cam.look_at({ 0, -0.1, 1 },
		    { 0, -0.1, -1 },
		    { 0, 1, 0 });

	float3 light_dir { 1, 1, 1 };
	normalize(light_dir);
	float light_dot_min = 0.95;
	float3 color_light { 0.6, 1.0, 1.0 };
	float3 color_sky { 0.72, 0.4, 0.3 };

	const uint n_samples = 16;
	float3 avg_color = {0, 0, 0};
	for (int i = 0; i < n_samples; ++i) {
		float3 color_mp { 1, 1, 1 };

		float2 rnd_pix_delta {curand_uniform(my_rnd), curand_uniform(my_rnd)};
		rnd_pix_delta = rnd_pix_delta * float2 {2.0f/cbuf_w, 2.0f/cbuf_h};
		float2 sample_pix = rnd_pix_delta + pix;
		Ray ray = cam.cast(sample_pix.x, sample_pix.y);
		HitRecord hit;

		int count = 100;
		uint sph_id = scene->n_sph;
		while (spheres_hit(scene->sph, scene->n_sph, &ray, &hit, &sph_id)) {
			color_mp = color_mp * hit.color;
			ray.o = ray.point_at(hit.t); //+ 0.001 * hit.n;
			ray.d = reflect(ray.d, hit.n);
			if (count-- == 0) {
				color_mp = { 0, 0, 1.0 };
				break;
			}
		}

		float3 color;
		if (dot(normalize(ray.d), light_dir) > light_dot_min)
			color = color_mp * color_light;
		else
			color = color_mp * color_sky;
		avg_color = avg_color + color;
	}
	avg_color = (1.0f / n_samples) * avg_color;
	cbuf[idx] = to_uchar4color(avg_color);
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
	curandState_t *rnd_buf;
	cudaMalloc(&rnd_buf, cbuf_s * sizeof(*rnd_buf));

	uint const block_dim = 256;
	uint const grid_dim = cbuf_s / block_dim;

	Scene scene_data, *scene_device;
	scene_data.n_sph = sizeof(defined_spheres) / sizeof(*defined_spheres);
	size_t sph_sz = sizeof(*scene_data.sph) * scene_data.n_sph;
	cudaMalloc(&scene_data.sph, sph_sz);
	cudaMalloc(&scene_device, sizeof(scene_data));

	setup_rnd_kernel<<<grid_dim, block_dim>>>(rnd_buf);
	cudaDeviceSynchronize();

	float phi = 2.35;
	while (1) {
		defined_spheres[1].o = defined_spheres[0].o + 0.70 * float3{sin(phi), cos(phi), sin(phi) / 4};
		phi += 0.005;
		if (phi > 2 * 3.1415)
			phi = 0;

		cudaMemcpy(scene_data.sph, defined_spheres, sph_sz, cudaMemcpyHostToDevice);
		cudaMemcpy(scene_device, &scene_data, sizeof(scene_data), cudaMemcpyHostToDevice);

		render<<<grid_dim, block_dim>>>(device_cbuf, fb.xres, fb.yres, scene_device, rnd_buf);
		cudaDeviceSynchronize();
		cudaMemcpy(fb.buf, device_cbuf, cbuf_s * sizeof(uchar4), cudaMemcpyDeviceToHost);
		fb.update();
	}
//	cudaFree(device_cbuf);
//	return 0;
}

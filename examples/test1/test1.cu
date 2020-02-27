#include <cuda_runtime.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <raytracer/camera.h>
#include <raytracer/geom.h>
#include <raytracer/fbuffer.h>
#include <raytracer/mouse.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

//#define LAMBERT_MODE

Sphere defined_spheres[] = {
	{.o = { 0, 0, -1 },
	 .r = 0.4,
	 .color = { 0.5, 0.4, 0.9 }},
	{.o = { 0.5, -0.5, -0.88 },
	 .r = 0.3,
	 .color = { 0.8, 0.4, 1.0 }},
	{.o = { 0.6, 0, -0.6 },
	 .r = 0.25,
	 .color = { 0.3, 1.0, 0.6 }},
	{.o = { -1.5, -0.7, -1.7 },
	 .r = 0.6,
	 .color = { 0.7, 0.8, 1.0 }},
	{.o = { 0, -0.6, -0.7 },
	 .r = 0.08,
	 .color = { 0.95, 0.95, 0.95 }},
	{.o = { 0, -0.4, -0.5 },
	 .r = 0.1,
	 .color = { 0.95, 0.95, 0.95 }},
};

struct Scene {
	Camera cam;
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

const uint render_n_samples = 1;
__global__
void render(uchar4 *cbuf, float3 *float_cbuf, uint cbuf_w, uint cbuf_h, Scene const *scene,
	curandState_t *rnd_buf, uint s_collected)
{
	uint const idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t *my_rnd = &rnd_buf[idx % blockDim.x];
	float2 const pix = 2.0f * float2{ float(idx % cbuf_w) / cbuf_w - 0.5f,
					  float(idx / cbuf_w) / cbuf_h - 0.5f};

	float3 light_dir = normalize(float3{ 1, 1, 1 });
	float const light_dot_min = 0.8;
	float3 color_light { 0.7, 1.0, 1.0 };
	float3 color_sky = 0.3 * float3 { 0.72, 0.4, 0.3 };

	float3 sum_color = {0, 0, 0};
	for (int i = 0; i < render_n_samples; ++i) {
		float3 color_mp { 1, 1, 1 };

		float2 rnd_pix_delta {curand_uniform(my_rnd), curand_uniform(my_rnd)};
		rnd_pix_delta = rnd_pix_delta * float2 {2.0f/cbuf_w, 2.0f/cbuf_h};
		float2 sample_pix = rnd_pix_delta + pix;
		Ray ray = scene->cam.cast(sample_pix.x, sample_pix.y);
		HitRecord hit;

		int count = 100;
		uint sph_id = scene->n_sph;
		while (spheres_hit(scene->sph, scene->n_sph, &ray, &hit, &sph_id)) {
			color_mp = color_mp * hit.color;
			ray.o = ray.point_at(hit.t); //+ 0.001 * hit.n;
#ifndef LAMBERT_MODE
			ray.d = reflect(ray.d, hit.n);
#else
			float a, b;
			a = curand_uniform(my_rnd) * 2 * 3.1415;
			b = asin(curand_uniform(my_rnd));
			ray.d = float3 {sin(b) * sin(a), sin(b) * cos(a), cos(b)};
#endif
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
		sum_color = sum_color + color;
	}
	sum_color = sum_color + float(s_collected) * float_cbuf[idx];
	sum_color = (1.0f/(render_n_samples + s_collected)) * sum_color;
	float_cbuf[idx] = sum_color;
	cbuf[idx] = to_uchar4color(sum_color);
}

int main()
{
	Fbuffer fb;
	Mouse ms;
	if (fb.init("/dev/fb0") < 0) {
		std::cerr << "Can't open /dev/fb0\n";
		return 1;
	}
	if (ms.init("/dev/input/mice") < 0) {
		std::cerr << "Can't open /dev/input/mice\n";
		return 1;
	}

	uchar4 *device_cbuf;
	float3 *float_cbuf;
	size_t const cbuf_s = fb.xres * fb.yres;
	cudaMalloc(&device_cbuf, cbuf_s * sizeof(uchar4));
	cudaMalloc(&float_cbuf, cbuf_s * sizeof(float3));
	curandState_t *rnd_buf;
	cudaMalloc(&rnd_buf, cbuf_s * sizeof(*rnd_buf));

	uint const block_dim = 256;
	uint const grid_dim = cbuf_s / block_dim;

	Scene scene_data, *scene_device;
	scene_data.n_sph = sizeof(defined_spheres) / sizeof(*defined_spheres);
	size_t sph_sz = sizeof(*scene_data.sph) * scene_data.n_sph;
	cudaMalloc(&scene_data.sph, sph_sz);
	cudaMalloc(&scene_device, sizeof(scene_data));

	scene_data.cam.set_screen(3.1415 / 6, float(fb.xres) / fb.yres);
	//scene_data.cam.look_at({0, -0.1, 1}, {0, -0.1, -1}, {0, 1, 0});

	setup_rnd_kernel<<<grid_dim, block_dim>>>(rnd_buf);
	cudaDeviceSynchronize();

	uint s_collected = 0;
	float2 rot = {0, 0};
	while (1) {
		Mouse::Event ev;
		if (ms.poll_ev(ev)) {
			rot = rot + 0.01 * float2 { float(ev.dx), float(ev.dy) };
			s_collected = 0;
		}
		float3 const cam_at {0, -0.1, -1};
		float3 const cam_up {0, 1, 0};
		float3 cam_eye = float3 {sin(rot.x) * sin(rot.y), sin(rot.x) * cos(rot.y), cos(rot.x)};
		cam_eye = 2 * cam_eye + cam_at;
		scene_data.cam.look_at(cam_eye, cam_at, cam_up);

		cudaMemcpy(scene_data.sph, defined_spheres, sph_sz, cudaMemcpyHostToDevice);
		cudaMemcpy(scene_device, &scene_data, sizeof(scene_data), cudaMemcpyHostToDevice);

		render<<<grid_dim, block_dim>>>(device_cbuf, float_cbuf, fb.xres, fb.yres, scene_device, rnd_buf, s_collected);
		s_collected += render_n_samples;
		cudaDeviceSynchronize();
		cudaMemcpy(fb.buf, device_cbuf, cbuf_s * sizeof(uchar4), cudaMemcpyDeviceToHost);
		fb.update();
	}
//	cudaFree(device_cbuf);
//	return 0;
}

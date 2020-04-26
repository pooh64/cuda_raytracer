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

#include <signal.h>

#define SIMPLE_BLUR_NFRAMES 0

#define eps_translate 0.001

#define __eval_device_fp(d_ptr)					\
({	typeof(d_ptr) h_ptr;					\
	cudaMemcpyFromSymbol(&h_ptr, d_ptr, sizeof(h_ptr));	\
	h_ptr;							\
})
template <typename T>
T eval_device_fp(T *d_ptr)
{
	return __eval_device_fp(*d_ptr);
}

/* material prototypes */
typedef void (*surface_func_t)(float3 &d, float3 const &n, curandState_t *rnd);
#define REGISTER_SURFACE(name)								\
__device__ surface_func_t surface_##name##_dptr = __surface_##name;			\
	   surface_func_t surface_##name##_hptr = eval_device_fp(&surface_##name##_dptr)

__device__
void __surface_lambert(float3 &d, float3 const &n, curandState_t *rnd)
{
	float tmp = rsqrt(n.x * n.x + n.y * n.y);
	float3 u = {n.y * tmp, -n.x * tmp, 0};
	float3 v = cross(u, n);

	float a, b;
	a = curand_uniform(rnd) * 2 * 3.1415;
	b = asin(curand_uniform(rnd));
	d = float3 {cos(b) * sin(a), cos(b) * cos(a), sin(b)};
	d = d.x * u + d.y * v + d.z * n;
}

__device__
void __surface_mirror(float3 &d, float3 const &n, curandState_t *rnd)
{
	d = reflect(d, n);
}

__device__
void __surface_glass(float3 &d, float3 const &n, curandState_t *rnd)
{
	float nt = 1.3;

	if (curand_uniform(rnd) < 0.1) {
		d = reflect(d, n);
		return;
	}

	float cos_val = dot(n, d);
	if (cos_val < 0) {
		float tmp = 1.0 / nt;
		cos_val = -cos_val;
		float root = 1.0 - (tmp * tmp) * (1.0 - cos_val * cos_val);
		d = tmp * d + (tmp * cos_val - sqrtf(root)) * n;
	} else {
		float tmp = dot(d, n);
		float root = 1.0 - (nt * nt) * (1.0 - tmp * tmp);
		if (root < 0) {
			d = reflect(d, n);
			return;
		}
		d = nt * d - (nt * tmp - sqrt(root)) * n;
	}
}

REGISTER_SURFACE(lambert);
REGISTER_SURFACE(mirror);
REGISTER_SURFACE(glass);

struct ShaderObject {
	surface_func_t surf;
	Sphere sph;
	float3 color;
};

ShaderObject defined_objects[] = {
	{.surf = surface_lambert_hptr,
	{.o = { 0, 0, -1 }, .r = 0.4},
	 .color = { 0.3, 0.3, 0.9 }},
	{.surf = surface_mirror_hptr,
	{.o = { 0.85, 0, -1 }, .r = 0.4},
	 .color = { 0.8, 0.8, 0.8 }},
	{.surf = surface_glass_hptr,
	{.o = { -0.85, 0, -1 }, .r = 0.4},
	 .color = { 0.9, 0.9, 0.9 }},
};

void screenshot_handler(int sig)
{
	system("fbcat /dev/fb0 >shot.pnm");
	exit(0);
}

void setup_screenshot(int sig)
{
	struct sigaction act;
	act.sa_handler = &screenshot_handler;
	sigaction(sig, &act, NULL);
}

struct Scene {
	Camera cam;
	ShaderObject *obj;
	uint  n_obj;
};

__device__
inline bool spheres_hit(ShaderObject *arr, uint arr_sz, Ray const *ray, HitRecord *hit,
	ShaderObject **obj)
{
	bool is_a_hit = false;
	float tmax = FLT_MAX;
	for (uint i = 0; i < arr_sz; ++i) {
		ShaderObject *ptr = &arr[i];
		if (ptr->sph.hit(*ray, 0, tmax, hit)) {
			is_a_hit = true;
			tmax = hit->t;
			*obj = ptr;
		}
	}
	return is_a_hit;
}

__device__
inline float2 rand_uniform_float2(curandState_t *rnd)
{
	return float2 {curand_uniform(rnd), curand_uniform(rnd)};
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
void render(uchar4 *cbuf, float3 *float_cbuf, uint cbuf_w, uint cbuf_h, Scene *scene,
	curandState_t *rnd_buf, uint s_collected)
{
	uint const idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState_t *my_rnd = &rnd_buf[idx % blockDim.x];
	float2 const pix = 2.0f * float2{ float(idx % cbuf_w) / cbuf_w - 0.5f,
					  float(idx / cbuf_w) / cbuf_h - 0.5f};

	float3 light_dir = normalize(float3{ 0, -0.2, 1 });
	float const light_dot_min = 0.85;
	float3 color_light { 1, 1, 1 };
	float3 color_sky = 0.2 / 255.0 * float3 { 98.0, 24.0, 19.0 };

	float3 sum_color = {0, 0, 0};
	for (int i = 0; i < render_n_samples; ++i) {
		float3 color_mp { 1, 1, 1 };

		float2 rnd_pix_delta {curand_uniform(my_rnd), curand_uniform(my_rnd)};
		rnd_pix_delta = rnd_pix_delta * float2 {2.0f/cbuf_w, 2.0f/cbuf_h};
		float2 sample_pix = rnd_pix_delta + pix;
		Ray ray = scene->cam.cast(sample_pix.x, sample_pix.y);
		HitRecord hit;

		int count = 100;
		ShaderObject *obj = NULL;
		while (spheres_hit(scene->obj, scene->n_obj, &ray, &hit, &obj)) {
			color_mp = color_mp * obj->color;
			ray.o = ray.point_at(hit.t);
			obj->surf(ray.d, hit.n, my_rnd);
			ray.o = ray.o + eps_translate * ray.d;

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

		color = sqrt(saturate(color));

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
	setup_screenshot(SIGINT);

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
	scene_data.n_obj = sizeof(defined_objects) / sizeof(*defined_objects);
	size_t obj_sz = sizeof(*scene_data.obj) * scene_data.n_obj;
	cudaMalloc(&scene_data.obj, obj_sz);
	cudaMalloc(&scene_device, sizeof(scene_data));

	scene_data.cam.set_screen(3.1415 / 4, float(fb.xres) / fb.yres);
	//scene_data.cam.look_at({0, -0.1, 1}, {0, -0.1, -1}, {0, 1, 0});

	setup_rnd_kernel<<<grid_dim, block_dim>>>(rnd_buf);
	cudaDeviceSynchronize();

	uint s_collected = 0;
	float2 rot = {0, 0};
	while (1) {
		Mouse::Event ev;
		if (ms.poll_ev(ev)) {
			rot = rot + 0.01 * float2 { -float(ev.dx), float(ev.dy) };
			s_collected = SIMPLE_BLUR_NFRAMES;
		}
		float3 const cam_at {0, 0, -1};
		float3 const cam_up {0, 1, 0};
		float3 cam_eye = float3 {cos(rot.y) * sin(rot.x), sin(rot.y), cos(rot.y) * cos(rot.x)};
		cam_eye = 2 * cam_eye + cam_at;
		scene_data.cam.look_at(cam_eye, cam_at, cam_up);

		// REMOVE
		cudaMemcpy(scene_data.obj, defined_objects, obj_sz, cudaMemcpyHostToDevice);
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

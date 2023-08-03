#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>

#include <cuda/std/complex>

#include "complex_vector.cu"
#include "sphere.h"
#include "rcs_params.h"

extern "C" {
	__constant__ Params params;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0,
	unsigned int& i1) {
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer(unsigned int i0,
	unsigned int i1) {
	const unsigned long long uptr =
		static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

struct Payload {
	unsigned int ray_id;  // unique id of the ray

	int refCount;

	float tpath;  // total path length until last bounes

	float3 polarization;

	float3 refNormal;
};

static __forceinline__ __device__ void trace(
	OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction,
	Payload* pld_ptr, int offset, int stride, int miss) {
	unsigned int p0, p1;
	packPointer(pld_ptr, p0, p1);
	float tmin = 1e-5f;
	float tmax = 1e30f;
	optixTrace(handle, ray_origin, ray_direction, tmin, tmax,
		0.0f,  // rayTime
		OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		offset,  // SBT offset
		stride,  // SBT stride(obj_count - 1)
		miss,    // missSBTIndex
		p0, p1);
}

static __forceinline__ __device__ Payload* getPayload() {
	unsigned int p0, p1;
	p0 = optixGetPayload_0();
	p1 = optixGetPayload_1();
	Payload* prd;
	prd = static_cast<Payload*>(unpackPointer(p0, p1));
	return prd;
}


// TODO: to improve performance, pre-compute and pack the normals.
__device__ __forceinline__ float3 getnormal(const unsigned int triId) {
	float3 vertex[3];
	OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
	optixGetTriangleVertexData(gas_handle, triId, 0, 0, vertex);

	float3 normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));

	return normal;
}

extern "C" __global__ void __raygen__rg() {
	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();

	float3 origin;
	float3 direction;
	int idR = idx.x;
	int idU = idx.y;

	origin = params.rayPosBegin + params.rayPosStepU * idU + params.rayPosStepR * idR;
	direction = -params.dirN;

	Payload pld;

	pld.polarization = params.polarization;
	pld.tpath = 0.0f;
	pld.ray_id = idx.x + dim.x * idx.y;
	pld.refCount = 0;

	Payload* pldptr = &pld;

	Result* result = params.result;
	result->ar_img = 0.0f;
	result->ar_real = 0.0f;
	result->au_img = 0.0f;
	result->au_real = 0.0f;
	result->refCount = 0;

	trace(params.handle, origin, direction, pldptr, 0, 1,
		0);
}

extern "C" __global__ void __miss__ms() {
	unsigned int p0, p1;
	Payload* pldptr = getPayload();
	packPointer(pldptr, p0, p1);
	float3 ray_direction = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	int ray_id = pldptr->ray_id;

	float c0 = 299792458.0f;
	float freq = params.freq;

	float angFreq = 2 * M_PIf * freq;
	float waveLen = c0 / freq;
	float waveNum = 2 * M_PIf / waveLen;

	float rayRadius = params.observer_pos.x / params.rays_per_dimension;
	float rayDiameter = rayRadius * 2;

	float rayArea = rayDiameter * rayDiameter;

	float phi = params.observer_pos.y;
	float the = params.observer_pos.z;

	float cp = cosf(phi);
	float sp = sinf(phi);
	float ct = cosf(the);
	float st = sinf(the);

	float3 dirX = make_float3(1.0, 0.0, 0.0);
	float3 dirY = make_float3(0.0, 1.0, 0.0);
	float3 dirZ = make_float3(0.0, 0.0, 1.0);
	float3 dirP = make_float3(-sp, cp, 0.0);
	float3 dirT = make_float3(cp * ct, sp * ct, -st);

	float3 vecK = waveNum * ((dirX * cp + dirY * sp) * st + dirZ * ct);

	using cuda::std::complex;
	complex<float> AU = 0;

	complex<float> AR = 0;
	complex<float> i = complex<float>(0.0f, 1.0f);

	if (pldptr->refCount > 0) {
		float kr = waveNum * pldptr->tpath;

		float reflectionCoef = powf(1.0f, pldptr->refCount);
		float3 pol = pldptr->polarization;

		complexFloat3 apE = exp(i * kr) * pol * reflectionCoef;

		complexFloat3 apH = -cross(apE, ray_direction);

		complex<float> BU =
			dot(-(cross(apE, -dirP) + cross(apH, dirT)), ray_direction);

		complex<float> BR =
			dot(-(cross(apE, dirT) + cross(apH, dirP)), ray_direction);

		float t = (waveNum * rayArea) / (4.0f * M_PIf);

		complex<float> e = exp(-i * dot(vecK, ray_ori));

		complex<float> factor = complex<float>(0.0, t) * e;


		AU = BU * factor;

		AR = BR * factor;
		/*if (pldptr->ray_id % 10000==0) {
			printf("factor: %f %f\n", factor.real(), factor.imag());
			printf("waveNum: %f\n", waveNum);
		}*/
	}

	params.result[ray_id].au_real = AU.real();
	params.result[ray_id].au_img = AU.imag();
	params.result[ray_id].ar_real = AR.real();
	params.result[ray_id].ar_img = AR.imag();

	params.result[ray_id].refCount = pldptr->refCount;
}

extern "C" __global__ void __closesthit__triangle() {
	unsigned int tri_id = optixGetPrimitiveIndex();

	float3 ray_dir = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	float ray_tmax = optixGetRayTmax();

	Payload* pldptr = getPayload();

	float3 out_normal = params.out_normals[tri_id];

	float3 hit_point = ray_ori + ray_tmax * ray_dir;
	float3 reflect_dir = reflect(ray_dir, out_normal);

	{
		float3 pol = pldptr->polarization;

		float3 hitNormal = out_normal;
		float3 dirCrossNormal = cross(ray_dir, hitNormal);

		float3 polU = normalize(dirCrossNormal);
		float3 polR = normalize(cross(ray_dir, polU));

		float3 refDir = reflect_dir;

		float3 refPolU = -polU;
		float3 refPolR = cross(refDir, refPolU);

		float polCompU = dot(pol, polU);
		float polCompR = dot(pol, polR);

		float total_path_length = ray_tmax + pldptr->tpath;
		pldptr->tpath = total_path_length;
		pldptr->polarization = -polCompR * refPolR + polCompU * refPolU;

		pldptr->refNormal = out_normal;
		pldptr->refCount += 1;
	}

	trace(params.handle, hit_point, reflect_dir, pldptr, 0, 1, 0);
}
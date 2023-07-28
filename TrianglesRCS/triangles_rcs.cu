#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>

#include <cuda/std/complex>

#include "complex_vector.cu"
#include "sphere.h"
#include "triangles_rcs.h"

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

static __forceinline__ __device__ void trace2(
	OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction,
	Payload* prd, int offset, int stride, int miss) {
	unsigned int p0, p1;
	packPointer(prd, p0, p1);
	float tmin2 = 1e-5f;
	float tmax2 = 1e30f;
	optixTrace(handle, ray_origin, ray_direction, tmin2, tmax2,
		0.0f,  // rayTime
		OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		offset,  // SBT offset
		stride,  // SBT stride(obj_count - 1)
		miss,    // missSBTIndex
		p0, p1);
}

static __forceinline__ __device__ Payload* getPayload2() {
	unsigned int p0, p1;
	p0 = optixGetPayload_0();
	p1 = optixGetPayload_1();
	Payload* prd;
	prd = static_cast<Payload*>(unpackPointer(p0, p1));
	return prd;
}

// (r, phi, theta) to (x, y, z) theta os the angle between z and x-y plane
static __forceinline__ __device__ float3 sphericalToCartesian(float3 point) {
	float r = point.x;
	float phi = point.y;
	float theta = point.z;
	float3 res = make_float3(r * sinf(theta) * cosf(phi),
		r * sinf(theta) * sinf(phi), r * cosf(theta));
	return res;
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim,
	float3& origin,
	float3& direction) {
	float3 outDirSph = params.cam_eye;
	float3 dirN = make_float3(0);  // ray direction
	float3 dirU = make_float3(0);
	float3 dirR = make_float3(0);

	float cp = cosf(outDirSph.y);
	float sp = sinf(outDirSph.y);
	float ct = cosf(outDirSph.z);
	float st = sinf(outDirSph.z);

	dirN.x = st * cp;
	dirN.y = st * sp;
	dirN.z = ct;

	dirR.x = sp;
	dirR.y = -cp;
	dirR.z = 0;

	dirU = cross(dirR, dirN);

	dirN = normalize(dirN);
	dirU = normalize(dirU);
	dirR = normalize(dirR);

	dirU = normalize(dirU - dot(dirU, dirN) * dirN);

	dirR = normalize(dirR - dot(dirR, dirN) * dirN);
	dirR = normalize(dirR - dot(dirR, dirU) * dirU);

	float3 boundBoxCenter = params.box_center;
	float boundBoxRadius = outDirSph.x;
	float3 rayPoolCenter = boundBoxCenter + dirN * 2.0 * boundBoxRadius;
	float3 rayPoolRectMin = rayPoolCenter - (dirR + dirU) * boundBoxRadius;
	float3 rayPoolRectMax = rayPoolCenter + (dirR + dirU) * boundBoxRadius;

	int rayCountSqrt = params.rays_per_dimension;

	float rayTubeRadius = boundBoxRadius / rayCountSqrt;
	float rayTubeDiameter = rayTubeRadius * 2.0f;
	float3 rayPosStepU = rayTubeDiameter * dirU;
	float3 rayPosStepR = rayTubeDiameter * dirR;
	float3 rayPosBegin = rayPoolRectMin + (rayPosStepU + rayPosStepR) / 2.0f;
	int idR = idx.x;
	int idU = idx.y;
	int idRay = idR + rayCountSqrt * idU;
	origin = rayPosBegin + rayPosStepU * idU + rayPosStepR * idR;
	direction = -dirN;
}

// TODO: to improve performance, pre-compute and pack the normals.
// but here we compute them while tracing
// available if BUILD_INPUT_TYPE TRIANGLES;
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

	float3 ray_origin;
	float3 ray_direction;

	computeRay(idx, dim, ray_origin, ray_direction);

	Payload pld;

	pld.polarization = make_float3(-0.5f, 0.5f, 0.0f);

	pld.tpath = 0.0f;
	pld.ray_id = idx.x + dim.x * idx.y;
	pld.refCount = 0;

	params.result[pld.ray_id].rid = pld.ray_id;

	Payload* pldptr = &pld;

	trace2(params.handle, ray_origin, ray_direction, pldptr, 0, 1,
		0);
}

extern "C" __global__ void __miss__ms() {
	unsigned int p0, p1;
	Payload* pldptr = getPayload2();
	packPointer(pldptr, p0, p1);
	// printf("miss ray: %d\n", pldptr->ray_id);
	float3 ray_direction = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	int ray_id = pldptr->ray_id;

	float c0 = 299792458.0;
	float freq = c0 * 10;

	float angFreq = 2 * M_PIf * freq;
	float waveLen = c0 / freq;
	float waveNum = 2 * M_PIf / waveLen;

	float rayRadius = params.cam_eye.x / params.rays_per_dimension;
	float rayDiameter = rayRadius * 2;

	float rayArea = rayDiameter * rayDiameter;

	float phi = params.cam_eye.y;
	float the = params.cam_eye.z;

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

		complexFloat3 apE = exp(i * kr) * pldptr->polarization;
		float3 pol = pldptr->polarization;

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

	Payload* pldptr = getPayload2();

	HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
	const MeshData* mesh_data = (MeshData*)data->shape_data;
	const uint3 index = mesh_data->indices[tri_id];

	const float3 v0 = mesh_data->vertices[index.x];
	const float3 v1 = mesh_data->vertices[index.y];
	const float3 v2 = mesh_data->vertices[index.z];
	const float3 out_normal = normalize(cross((v1 - v0), (v2 - v0)));

	float3 hit_point = ray_ori + ray_tmax * ray_dir;
	float3 reflect_dir = reflect(ray_dir, out_normal);

	float tmin = 1e-5f;
	float tmax = 1e30f;

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

	trace2(params.handle, hit_point, reflect_dir, pldptr, 0, 1, 0);
}

extern "C" __global__ void __closesthit__sphere() {
	Payload* pldptr = getPayload2();
	unsigned int sphe_id = optixGetPrimitiveIndex();

	float ray_tmax = optixGetRayTmax();
	float total_path_length = ray_tmax + pldptr->tpath;

	pldptr->tpath = total_path_length;
}

extern "C" __global__ void __intersection__sphere() {
	HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

	const int prim_idx = optixGetPrimitiveIndex();
	const SphereData sphere_data = ((SphereData*)data->shape_data)[prim_idx];

	const float3 center = sphere_data.center;
	const float radius = sphere_data.radius;

	const float3 origin = optixGetObjectRayOrigin();
	const float3 direction = optixGetObjectRayDirection();

	Payload* pldptr = getPayload2();
	float ray_tmax = optixGetRayTmax();

	float total_path_length = ray_tmax + pldptr->tpath;
	pldptr->tpath = total_path_length;
	// float result = ((1.0f * 1.0f * 1.0f) / (16.0f * M_PIf * M_PIf)) *
	//                (pldptr->refIdx / total_path_length);
	params.result[pldptr->ray_id].rid = pldptr->ray_id;
}

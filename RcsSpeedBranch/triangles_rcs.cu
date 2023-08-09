#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include <cuda/std/complex>

#include "complex_vector.cu"
#include "rcs_params.h"

extern "C" {
	__constant__ Params params;
}



static __forceinline__ __device__ void trace(
	OptixTraversableHandle handle, float3 ray_origin, float3 ray_direction,
	int offset, int stride, int miss, unsigned int relCount, float tpath, float3 pol) {

	float tmin = 1e-5f;
	float tmax = 1e30f;
	unsigned path_uint = __float_as_uint(tpath);
	unsigned pol1 = __float_as_uint(pol.x);
	unsigned pol2 = __float_as_uint(pol.y);
	unsigned pol3 = __float_as_uint(pol.z);
	/*unsigned norm1 = __float_as_uint(out_normals.x);
	unsigned norm2 = __float_as_uint(out_normals.y);
	unsigned norm3 = __float_as_uint(out_normals.z);*/

	optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT, offset, stride, miss, relCount, path_uint, pol1, pol2, pol3);
}


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
	int ray_id = idx.x + dim.x * idx.y;

	float3 origin;
	float3 direction;
	int idR = idx.x;
	int idU = idx.y;

	origin = params.rayPosBegin + params.rayPosStepU * idU + params.rayPosStepR * idR;
	direction = params.rayDir;

	trace(params.handle, origin, direction, 0, 1,
		0, 0u, 0.0f, params.polarization);
}



extern "C" __global__ void __miss__ms() {
	
	float3 ray_direction = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	const uint3 idx = optixGetLaunchIndex();
	const uint3 dim = optixGetLaunchDimensions();
	int ray_id = idx.x + dim.x * idx.y;

	float wave_num = params.wave_num;

	float phi = params.observer_angle.x;
	float the = params.observer_angle.y;

	float cp = cosf(phi);
	float sp = sinf(phi);
	float ct = cosf(the);
	float st = sinf(the);

	float3 dirX = make_float3(1.0, 0.0, 0.0);
	float3 dirY = make_float3(0.0, 1.0, 0.0);
	float3 dirZ = make_float3(0.0, 0.0, 1.0);
	float3 dirP = make_float3(-sp, cp, 0.0);
	float3 dirT = make_float3(cp * ct, sp * ct, -st);

	float3 vecK = wave_num * ((dirX * cp + dirY * sp) * st + dirZ * ct);

	using cuda::std::complex;
	complex<float> AU = 0;

	complex<float> AR = 0;
	complex<float> i = complex<float>(0.0f, 1.0f);
	float t_value = params.t_value;
	unsigned int refCount = optixGetPayload_0();
	if (refCount > 0) {
		//float kr = wave_num * pldptr->tpath;
		float tpath = __uint_as_float(optixGetPayload_1());
		float kr = wave_num * tpath;
		float3 pol;
		pol.x = __uint_as_float(optixGetPayload_2());
		pol.y = __uint_as_float(optixGetPayload_3());
		pol.z = __uint_as_float(optixGetPayload_4());


		float relectance = params.reflectance;
		float reflectionCoef = powf(relectance, refCount);
		//pol = pldptr->polarization;

		complexFloat3 apE = exp(i * kr) * pol * reflectionCoef;

		complexFloat3 apH = -cross(apE, ray_direction);

		complex<float> BU =
			dot(-(cross(apE, -dirP) + cross(apH, dirT)), ray_direction);

		complex<float> BR =
			dot(-(cross(apE, dirT) + cross(apH, dirP)), ray_direction);

		complex<float> e = exp(-i * dot(vecK, ray_ori));

		complex<float> factor = complex<float>(0.0, t_value) * e;


		AU = BU * factor;

		AR = BR * factor;
		/*if (pldptr->ray_id % 10000==0) {
			printf("factor: %f %f\n", factor.real(), factor.imag());
			printf("waveNum: %f\n", waveNum);
		}*/
	}
	params.result[4 * ray_id] = AU.real();
	params.result[4 * ray_id + 1] = AU.imag();
	params.result[4 * ray_id + 2] = AR.real();
	params.result[4 * ray_id + 3] = AR.imag();
}



extern "C" __global__ void __closesthit__triangle() {
	unsigned int tri_id = optixGetPrimitiveIndex();

	float3 ray_dir = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	float ray_tmax = optixGetRayTmax();

	//Payload* pldptr = getPayload();

	float3 out_normal = params.out_normals[tri_id];

	float3 hit_point = ray_ori + ray_tmax * ray_dir;
	float3 reflect_dir = reflect(ray_dir, out_normal);

	float tpath = __uint_as_float(optixGetPayload_1());

	float3 pol;
	pol.x = __uint_as_float(optixGetPayload_2());
	pol.y = __uint_as_float(optixGetPayload_3());
	pol.z = __uint_as_float(optixGetPayload_4());

	float3 hitNormal = out_normal;
	float3 dirCrossNormal = cross(ray_dir, hitNormal);

	float3 polU = normalize(dirCrossNormal);
	float3 polR = normalize(cross(ray_dir, polU));

	float3 refDir = reflect_dir;

	float3 refPolU = -polU;
	float3 refPolR = cross(refDir, refPolU);

	float polCompU = dot(pol, polU);
	float polCompR = dot(pol, polR);

	float total_path_length = ray_tmax + tpath;
	/*pldptr->tpath = total_path_length;*/
	pol = -polCompR * refPolR + polCompU * refPolU;
	//pldptr->polarization = pol;

	//pldptr->refNormal = out_normal;
	unsigned int refCount = optixGetPayload_0() + 1u;
	//pldptr->refCount = refCount;

	trace(params.handle, hit_point, reflect_dir, /*pldptr,*/ 0, 1, 0, refCount, total_path_length, pol);
}
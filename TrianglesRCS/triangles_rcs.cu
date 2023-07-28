// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
#include <optix.h>

#include "triangles_rcs.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

#include "sphere.h"
#include "complex_vector.cu"
#include <cuda/std/complex>
#include <cuda_runtime.h>
//#include <cuComplex.h>

#define float3_as_ints( u ) float_as_int( u.x ), float_as_int( u.y ), float_as_int( u.z )

extern "C" {
	__constant__ Params params;
}


static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}


struct Payload {
	//LUV::Vec3< T > pos_ = float3 ray_ori = optixGetWorldRayOrigin();
	//LUV::Vec3< T > dir_ = float3 ray_dir = optixGetWorldRayDirection();

	unsigned int ray_id; // unique id of the ray

	//U32 refCount_
	int refCount;

	//U32 lastHitIdx_;
	//int lastHitIdx;

	//T dist_ = float tpath
	float tpath;         // total lenth of the path with multiple bounces until last bounes
	float refIdx;
	//float power;

	//LUV::Vec3< T > pol_;
	float3 polarization;

	//LUV::Vec3< T > refNormal_;
	float3 refNormal;

};

static __forceinline__ __device__ void trace2(
	OptixTraversableHandle handle,
	float3                 ray_origin,
	float3                 ray_direction,
	float                  tmin,
	float                  tmax,
	Payload* prd,
	int                    offset,
	int                    stride,
	int                    miss
)
{
	unsigned int p0, p1;
	packPointer(prd, p0, p1);
	tmax = 1e20;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		tmin,
		tmax,
		0.0f,                // rayTime
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		offset,                   // SBT offset
		stride,                   // SBT stride(obj_count - 1)
		miss,                     // missSBTIndex
		p0, p1);
}


static __forceinline__ __device__ Payload* getPayload2()
{
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
	float3 res = make_float3(r * sin(theta) * cos(phi), r * sin(theta) * sin(phi), r * cos(theta));
	return res;
}

static __forceinline__ __device__ void computeRay(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	/*
	* 	// with an elevation angle ¦È
	float phi = 45;
	float theta = 30 * M_PI / 180.0; // radian of elevation
	float range = distance * 1000.0;
	//float z = range * sin(theta);
	//float r = range * cos(theta);
	phi = phi * M_PI / 180.0; // radian of phi
	//float x = r * sin(phi);
	//float y = r * cos(phi);
	*/
	// origin in Cartesian

	float3 ori = sphericalToCartesian(params.cam_eye);
	float theta = static_cast<float>(idx.x) / static_cast<float>(dim.x);
	float phi = static_cast<float>(idx.y) / static_cast<float>(dim.y);
	float ele = M_PIf * theta;
	float azi = 2.0f * M_PIf * phi;
	float radius = 1.0f;
	direction = make_float3(sinf(ele) * cosf(azi), sinf(ele) * sinf(azi), cosf(ele));
	origin = ori + direction * radius;
	//origin = params.cam_eye;
	//printf("ray origin: %f, %f, %f\n", origin.x, origin.y, origin.z);
}

//static __device__ float3 findOrthogonalVector(const float3& v) {
//	// Start with a unit vector along the x-axis
//	float3 unitX = { 1, 0, 0 };
//
//	// If v is not parallel to the x-axis, return the cross product of v and the
//	// x unit vector
//	if (v.x != v.y || v.x != v.z) {
//		return cross(v, unitX);
//	}
//
//	// Otherwise, use the y-axis or the z-axis
//	float3 unitY = { 0, 1, 0 };
//	return cross(v, unitY);
//}

static __forceinline__ __device__ void computeRay2(uint3 idx, uint3 dim, float3& origin, float3& direction)
{
	float3 ori = sphericalToCartesian(params.cam_eye);
	float theta = static_cast<float>(idx.x) / static_cast<float>(dim.x);
	float phi = static_cast<float>(idx.y) / static_cast<float>(dim.y);
	float ele = M_PIf * theta;
	float azi = 2.0f * M_PIf * phi;
	float radius = params.cam_eye.x;
	origin = make_float3(radius * sinf(ele) * cosf(azi), radius * sinf(ele) * sinf(azi), radius * cosf(ele));
	direction = -origin;

}

static __forceinline__ __device__ void computeRay3(uint3 idx, uint3 dim, float3& origin, float3& direction) {

	float3 outDirSph = params.cam_eye;
	float3 dirN = make_float3(0);  // ray direction
	float3 dirU = make_float3(0);
	float3 dirR = make_float3(0);
	//printf("start\n");
	//OrthonormalSet(outDirSph.y, outDirSph.z, dirN, dirU, dirR);

	float cp = cosf(outDirSph.y);
	float sp = sinf(outDirSph.y);
	float ct = cosf(outDirSph.z);
	float st = sinf(outDirSph.z);

	dirN.x = st * cp;
	dirN.y = st * sp;
	dirN.z = ct;

	/*if (idx.x % 1000 == 0 && idx.y%1000==0) {
		printf("ray direction: %f, %f, %f\n", dirN.x, dirN.y, dirN.z);
		printf("phi: %f, theta: %f\n", params.cam_eye.y, params.cam_eye.z);

	}*/

	dirR.x = sp;
	dirR.y = -cp;
	dirR.z = 0;

	dirU = cross(dirR, dirN);

	//Orthonormalize(dirN, dirU, dirR);

	dirN = normalize(dirN);
	dirU = normalize(dirU);
	dirR = normalize(dirR);

	dirU = normalize(dirU - dot(dirU, dirN) * dirN);

	dirR = normalize(dirR - dot(dirR, dirN) * dirN);
	dirR = normalize(dirR - dot(dirR, dirU) * dirU);

	//printf("normalized\n");

	float3 boundBoxCenter = params.boxCenter;
	//float boundBoxRadius = 0.866025f;
	float boundBoxRadius = outDirSph.x;
	float3 rayPoolCenter =
		boundBoxCenter + dirN * 2.0 * boundBoxRadius;
	//printf("rayPoolCenter %f, %f, %f\n", rayPoolCenter.x, rayPoolCenter.y, rayPoolCenter.z);

	float3 rayPoolRectMin =
		rayPoolCenter - (dirR + dirU) * boundBoxRadius;
	float3 rayPoolRectMax =
		rayPoolCenter + (dirR + dirU) * boundBoxRadius;

	int rayCountSqrt = params.image_height;

	float rayTubeRadius = boundBoxRadius / rayCountSqrt;
	float rayTubeDiameter = rayTubeRadius * 2.0f;
	float3 rayPosStepU = rayTubeDiameter * dirU;
	float3 rayPosStepR = rayTubeDiameter * dirR;
	float3 rayPosBegin =
		rayPoolRectMin + (rayPosStepU + rayPosStepR) / 2.0f;
	int idR = idx.x;
	int idU = idx.y;
	int idRay = idR + rayCountSqrt * idU;
	origin =
		rayPosBegin + rayPosStepU * idU + rayPosStepR * idR;
	direction = -dirN;
	//printf("generated\n");
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



	//float dx = static_cast<float>(idx.x) / static_cast<float>(dim.x);
	//float dy = static_cast<float>(idx.y) / static_cast<float>(dim.y);

	//create a ray sphere for rwpl
	float3 ray_origin;
	float3 ray_direction;


	computeRay3(idx, dim, ray_origin, ray_direction);
	//printf("idx: %d %d %d\n", idx.x, idx.y, idx.z);


	//printf("ray_length = %f\n", dot(ray_direction, ray_direction));

	// setting the per ray data (payload)
	Payload pld;
	/*
	*
		// rayArray[ idRay ].pos_ = rayPosBegin + rayPosStepU * (T)idU + rayPosStepR * (T)idR;
		// rayArray[ idRay ].dir_ = dirN;
		TODO rayArray[ idRay ].pol_ = polDir;
		// rayArray[ idRay ].dist_ = 1E36;
		rayArray[ idRay ].refCount_ = 0;
		// rayArray[ idRay ].lastHitIdx_ = -1;
	*/
	pld.polarization = make_float3(-0.5f, 0.5f, 0.0f);
	//pld.polarization = make_float3(0.0f, 0.0f, 1.0f);
	pld.tpath = 0.0f;
	pld.ray_id = idx.x + dim.x * idx.y;
	pld.refIdx = 0.0f;
	pld.refCount = 0;

	params.result[pld.ray_id].rid = pld.ray_id;

	Payload* pldptr = &pld;
	//printf("%d, %d, %d, %d = %d\n", idx.x, idx.y, dim.x, dim.y, );
	//if (pld.ray_id % 10000 == 0) {
	//	printf("ray direction: %f, %f, %f\n", ray_direction.x, ray_direction.y, ray_direction.z);
	//	printf("phi: %f, theta: %f\n", params.cam_eye.y, params.cam_eye.z);

	//}
	//printf("ray origin %f, %f, %f\n", ray_origin.x, ray_origin.y, ray_origin.z);


	float tmin = 1e-10f;
	float tmax = 1e30f;

	trace2(params.handle,
		ray_origin,
		ray_direction,
		tmin,  // tmin
		tmax,  // tmax
		pldptr,
		0,
		1,
		0);
	//Result* idata = params.result;
}


extern "C" __global__ void __miss__ms()
{
	unsigned int p0, p1;
	Payload* pldptr = getPayload2();
	packPointer(pldptr, p0, p1);
	//printf("miss ray: %d\n", pldptr->ray_id);
	float3 ray_direction = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();

	int ray_id = pldptr->ray_id;
	//printf("miss  ray: %d, direction vec(%f, %f, %f)\n", ray_id, ray_ori.x, ray_ori.y, ray_ori.z);
	//params.result[ray_id].rid = ray_id;


	float c0 = 299792458.0;
	float freq = c0 * 10;

	float angFreq = 2 * M_PIf * freq;
	float waveLen = c0 / freq;
	float waveNum = 2 * M_PIf / waveLen;

	//float radius = params.cam_eye.x / params.image_height;
	// assume 1 ray per area
	//int rayCount = params.image_height * params.image_height;
	//float rayArea = 4 * M_PIf * radius * radius;
	float rayRadius = params.cam_eye.x / params.image_height;
	float rayDiameter = rayRadius * 2;
	//printf("D: %f\n", rayDiameter);
	//printf("bound radius: %f\n", params.cam_eye.x);
	float rayArea = rayDiameter * rayDiameter;

	//float3 obsDir = params.cam_eye;
	//float3 obsDirSph = CtsToSph(obsDir);

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

	//printf("waveNum: %f\n", waveNum);
	float3 vecK =
		waveNum * ((dirX * cp + dirY * sp) * st + dirZ * ct);
	//printf("vecK %f, %f, %f\n", vecK.x, vecK.y, vecK.z);

	using cuda::std::complex;
	complex<float> AU = 0;

	complex<float> AR = 0;
	complex<float> i = complex<float>(0.0f, 1.0f);


	if (pldptr->refCount > 0) {
		//printf("Triangle hit Ray = %d, pathlen = %f\n", pldptr->ray_id, pldptr->tpath);

		float kr = waveNum * pldptr->tpath;
		//printf("kr: %f\n", kr);
		//printf("distance : %f\n", pldptr->tpath);

		float reflectionCoef = powf(1.0f, pldptr->refCount);
		//printf("reflectionCoef: %f\n", reflectionCoef);

		//auto ep = exp(i * kr);
		//printf("exp: %f + %f i\n", ep.real(), ep.imag());

		//complexFloat3 apE =
		//	exp(i * kr) * pldptr->polarization * reflectionCoef;
		complexFloat3 apE =
			exp(i * kr) * pldptr->polarization;
		float3 pol = pldptr->polarization;
		//printf("pol: %f, %f, %f\n", pol.x, pol.y, pol.z);

		//printComplexFloat3(apE);

		complexFloat3 apH = -cross(apE, ray_direction);

		complex<float> BU =
			dot(-(cross(apE, -dirP) + cross(apH, dirT)),
				ray_direction);
		//printf("dirp: %f, %f, %f\n", dirP.x, dirP.y, dirP.z);
		//printf("dirp: %f, %f, %f\n", dirP.x, dirP.y, dirP.z);
		//printComplexFloat3(apE);
		complex<float> BR = dot(
			-(cross(apE, dirT) + cross(apH, dirP)), ray_direction);

		float t = (waveNum * rayArea) / (4.0f * M_PIf);
		//printf("t: %f\n", t);

		complex<float> e = exp(-i * dot(vecK, ray_ori));
		//printf("e: %f + %f i\n", e.real, e.img);


		complex<float> factor = complex<float>(0.0, t) * e;
		//printf("factor: %f + %f i\n", factor.real(), factor.imag());

		AU = BU * factor;
		//printf("AU: %f + %f i\n", AU.real, AU.img);
		//printf("BU: %f + %f i\n", BU.real(), BU.imag());
		//printf("BR: %f + %f i\n", BR.real(), BR.imag());

		AR = BR * factor;
		//printf(" reflected miss ray: %d\n", params.result[ray_id].rid);

	}


	params.result[ray_id].au_real = AU.real();
	params.result[ray_id].au_img = AU.imag();
	params.result[ray_id].ar_real = AR.real();
	params.result[ray_id].ar_img = AR.imag();
	params.result[ray_id].refCount = pldptr->refCount;

}


extern "C" __global__ void __closesthit__triangle()
{
	unsigned int tri_id = optixGetPrimitiveIndex();
	//unsigned int sbt_id = optixGetSbtGASIndex();
	//float time = optixGetRayTime();
	//printf("tri[%d] = sbt[%d]\n", tri_id, sbt_id);

	float3 ray_dir = optixGetWorldRayDirection();
	float3 ray_ori = optixGetWorldRayOrigin();
	//printf("dir = (%f, %f, %f)\n", ray_dir.x, ray_dir.y, ray_dir.z);
	/*
	const float3 out_normal =
		  make_float3(
				int_as_float( optixGetAttribute_0() ),
				int_as_float( optixGetAttribute_1() ),
				int_as_float( optixGetAttribute_2() )
				);

	float3 vertex[3];
	OptixTraversableHandle gas_handle = optixGetGASTraversableHandle();
	optixGetTriangleVertexData(gas_handle, tri_id, sbt_id, time, vertex);

	float3 out_normal = cross((vertex[1] - vertex[0]), (vertex[2] - vertex[0]));
	*/

	// printf("prim[%d] = vec(%f, %f, %f)\n", sbt_id, out_normal.x, out_normal.y, out_normal.z);

	// We defined out geometry as a triangle geometry. In this case the
	// We add the t value of the intersection
	float ray_tmax = optixGetRayTmax();

	Payload* pldptr = getPayload2();


	// report individual bounces
	//printf("Ray = %d, pathlen = %f\n", pldptr->ray_id, total_path_length);


	//get vertice data from SBT and compute normal
	HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();
	const MeshData* mesh_data = (MeshData*)data->shape_data;
	const uint3 index = mesh_data->indices[tri_id];

	const float3 v0 = mesh_data->vertices[index.x];
	const float3 v1 = mesh_data->vertices[index.y];
	const float3 v2 = mesh_data->vertices[index.z];
	const float3 out_normal = normalize(cross((v1 - v0), (v2 - v0)));

	float3 hit_point = ray_ori + ray_tmax * ray_dir;
	float3 reflect_dir = reflect(ray_dir, out_normal);
	//printf("triangle vertex =(%f,%f,%f), (%f,%f,%f), (%f,%f,%f)\n", v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
	//printf("vec(%f, %f, %f)\n", reflect_dir.x, reflect_dir.y, reflect_dir.z);

	//// cos1
	//float cos1 = -1.0f * dot(ray_dir, out_normal) / (length(ray_dir) * length(out_normal));

	////cos2
	//float n_ij = 1.5f;
	//float cos2 = sqrtf((n_ij * n_ij) - (1.0f - cos1 * cos1)) / n_ij;
	////printf("cos2 = %f\n", cos2);

	////myu
	//float u1 = 1.0f;
	//float u2 = 2.0f;

	////R
	////float Rp = (u1 * n_ij * cos1 - u2 * cos2)/(u1 * n_ij * n_ij * cos1 + u2 * cos2);
	//float Rv = (u2 * n_ij * cos1 - u1 * cos2) / (u2 * n_ij * n_ij * cos1 + u1 * cos2);

	//float Rv_total = pldptr->refIdx + Rv;
	//pldptr->refIdx = Rv_total;

	// Minimal distance the ray has to travel to report next hit
	float tmin = 1e-5;
	float tmax = 1e40;

	{
		float3 pol = pldptr->polarization;

		float3 hitNormal = out_normal;
		float3 dirCrossNormal = cross(ray_dir, hitNormal);

		float3 polU = normalize(dirCrossNormal);
		float3 polR =
			normalize(cross(ray_dir, polU));

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
		//pldptr->lastHitIdx = hitIdx;
	}


	trace2(params.handle,
		hit_point,
		reflect_dir,
		tmin,  // tmin
		tmax,  // tmax
		pldptr,
		0,
		1,
		0);

	//printf("Triangle hit Ray = %d, pathlen = %f, ref_idx = %f\n", pldptr->ray_id, pldptr->tpath, pldptr->refIdx);
}

extern "C" __global__ void __closesthit__sphere()
{
	Payload* pldptr = getPayload2();
	unsigned int sphe_id = optixGetPrimitiveIndex();

	// We defined out geometry as a triangle geometry. In this case the
	// We add the t value of the intersection
	float ray_tmax = optixGetRayTmax();

	//float4 payload = getPayload();
	float total_path_length = ray_tmax + pldptr->tpath;
	//float total_path_length = ray_tmax + payload.y;

	if (pldptr->refIdx == 0.0f) {
		pldptr->refIdx = 1.0f;
	}


	//float result = ((1.0 * 1.0 * 1.0) / (16 * M_PIf * M_PIf)) * (payload.z/total_path_length);
	//float result = ((1.0 * 1.0 * 1.0) / (16 * M_PIf * M_PIf)) * (pldptr->refIdx / total_path_length);

	pldptr->tpath = total_path_length;
	//pldptr->power = result;
	//optixSetPayload_1(__float_as_uint(total_path_length));
	//optixSetPayload_3(__float_as_uint(result));

	//float* output = params.result;
	//atomicAdd(output + sphe_id, result);

	//params.result[prdptr->ray_id] = result;

	//printf("Sphe[%d], result = %f\n", sphe_id, pldptr->receive);
	//printf("Sphe[%d], result = %f\n", sphe_id, params.result[sphe_id]);
	//printf("Sphe[%d], result = %f\n", sphe_id, total_path_length);
	//printf("ray[%d], result = %f\n", pldptr->ray_id, pldptr->receive);
	//printf("%d\n", pldptr->ray_id);
	//printf("%f\n", pldptr->receive);

	// checking for debug
	/*
	if (result < 0.005f) {
	  printf("Ray = %d, pathlen = %f\n", __float_as_uint(payload.x), result);
	  printf("Sphe[%d], result = %f\n", sphe_id, params.result[sphe_id]);
	}
	*/
	//printf("Sphe[%d], result = %f\n", sphe_id, params.result[sphe_id]);
	//printf("closehit sphere\n");

}


extern "C" __global__ void __intersection__sphere()
{
	HitGroupData* data = (HitGroupData*)optixGetSbtDataPointer();

	const int prim_idx = optixGetPrimitiveIndex();
	const SphereData sphere_data = ((SphereData*)data->shape_data)[prim_idx];

	const float3 center = sphere_data.center;
	const float radius = sphere_data.radius;

	float err = 0.0001;
	const float3 origin = optixGetObjectRayOrigin();
	const float3 direction = optixGetObjectRayDirection();

	if ((origin.x - center.x) <= err && (origin.y - center.y) <= err && (origin.z - center.z) <= err) {
		//printf("penetrating\n");
		return;
	}
	//printf("intersection sphere, origin: %f, %f, %f, center: %f, %f, %f\n", origin.x, origin.y, origin.z, center.x, center.y, center.z);
	Payload* pldptr = getPayload2();
	float ray_tmax = optixGetRayTmax();

	float total_path_length = ray_tmax + pldptr->tpath;
	pldptr->tpath = total_path_length;
	float result = ((1.0f * 1.0f * 1.0f) / (16.0f * M_PIf * M_PIf)) * (pldptr->refIdx / total_path_length);
	params.result[pldptr->ray_id].rid = pldptr->ray_id;
	/*
	const float tmin = optixGetRayTmin();
	const float tmax = optixGetRayTmax();


	const float3 oc = origin - center;
	const float a = dot(direction, direction);
	const float half_b = dot(oc, direction);
	const float c = dot(oc, oc) - radius * radius;

	const float discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return;

	const float sqrtd = sqrtf(discriminant);

	float root = (-half_b - sqrtd) / a;
	if (root < tmin || tmax < root)
	{
		root = (-half_b + sqrtd) / a;
		if (root < tmin || tmax < root)
			return;
	}


	const float3 P = origin + root * direction;
	const float3 normal = (P - center) / radius;


	float phi = atan2(normal.y, normal.x);
	if (phi < 0) phi += 2.0f * M_PIf;
	const float theta = acosf(normal.z);
	const float2 texcoord = make_float2(phi / (2.0f * M_PIf), theta / M_PIf);


	optixReportIntersection(root, 0,
		__float_as_int(normal.x), __float_as_int(normal.y), __float_as_int(normal.z),
		__float_as_int(texcoord.x), __float_as_int(texcoord.y)
	);
	*/
}


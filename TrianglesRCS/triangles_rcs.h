#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include <fstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "TrianglesRCS/reduce.cu"
#include "rcs_params.h"

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#endif

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

#ifndef TRIANGLES_RCS_
#define TRIANGLES_RCS_

template <typename T>
struct SbtRecord {
	__align__(
		OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


static void context_log_cb(unsigned int level, const char* tag,
	const char* message, void* /*cbdata */) {
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
		<< "]: " << message << "\n";
}

struct GeometryAccelData {
	OptixTraversableHandle handle;
	CUdeviceptr d_output_buffer;
	uint32_t num_sbt_records;
};

// Instance acceleration structure
//
struct InstanceAccelData {
	OptixTraversableHandle handle;
	CUdeviceptr d_output_buffer;

	CUdeviceptr d_instances_buffer;
};

// -----------------------------------------------------------------------
// Geometry acceleration structure
// -----------------------------------------------------------------------
void buildGAS(OptixDeviceContext context, GeometryAccelData& gas,
	OptixBuildInput& build_input) {
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes gas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
		&build_input,
		1,  // Number of build inputs
		&gas_buffer_sizes));

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer),
		gas_buffer_sizes.tempSizeInBytes));

	CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
	size_t compacted_size_offset =
		roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
		compacted_size_offset + 8));

	OptixAccelEmitDesc emit_property = {};
	emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_property.result =
		(CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size +
			compacted_size_offset);

	OPTIX_CHECK(optixAccelBuild(context,
		0,  // CUDA stream
		&accel_options, &build_input,
		1,  // num build inputs
		d_temp_buffer, gas_buffer_sizes.tempSizeInBytes,
		d_buffer_temp_output_gas_and_compacted_size,
		gas_buffer_sizes.outputSizeInBytes, &gas.handle,
		&emit_property,  // emitted property list
		1                // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emit_property.result,
		sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas.d_output_buffer),
			compacted_gas_size));
		OPTIX_CHECK(optixAccelCompact(context, 0, gas.handle,
			gas.d_output_buffer, compacted_gas_size,
			&gas.handle));
		CUDA_CHECK(
			cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
	}
	else {
		gas.d_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
	}
}

OptixAabb sphereBound(const SphereData& sphere) {
	const float3 center = sphere.center;
	const float radius = sphere.radius;
	return OptixAabb{/* minX = */ center.x - radius,
		/* minY = */ center.y - radius,
		/* minZ = */ center.z - radius,
		/* maxX = */ center.x + radius,
		/* maxY = */ center.y + radius,
		/* maxZ = */ center.z + radius };
}

uint32_t getNumSbtRecords(const std::vector<uint32_t>& sbt_indices) {
	std::vector<uint32_t> sbt_counter;
	for (const uint32_t& sbt_idx : sbt_indices) {
		auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
		if (sbt_counter.empty() || itr == sbt_counter.end())
			sbt_counter.emplace_back(sbt_idx);
	}
	return static_cast<uint32_t>(sbt_counter.size());
}

void* buildSphereGAS(OptixDeviceContext context, GeometryAccelData& gas,
	const std::vector<SphereData>& spheres,
	const std::vector<uint32_t>& sbt_indices) {
	std::vector<OptixAabb> aabb;
	std::transform(
		spheres.begin(), spheres.end(), std::back_inserter(aabb),
		[](const SphereData& sphere) { return sphereBound(sphere); });

	CUdeviceptr d_aabb_buffer;
	const size_t aabb_size = sizeof(OptixAabb) * aabb.size();
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_buffer), aabb.data(),
		aabb_size, cudaMemcpyHostToDevice));

	CUdeviceptr d_sbt_indices;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices),
		sizeof(uint32_t) * sbt_indices.size()));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_sbt_indices), sbt_indices.data(),
		sizeof(uint32_t) * sbt_indices.size(), cudaMemcpyHostToDevice));

	void* d_sphere;
	CUDA_CHECK(cudaMalloc(&d_sphere, sizeof(SphereData) * spheres.size()));
	CUDA_CHECK(cudaMemcpy(d_sphere, spheres.data(),
		sizeof(SphereData) * spheres.size(),
		cudaMemcpyHostToDevice));

	uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
	gas.num_sbt_records = num_sbt_records;

	uint32_t* input_flags = new uint32_t[num_sbt_records];
	for (uint32_t i = 0; i < num_sbt_records; i++)
		input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

	OptixBuildInput sphere_input = {};
	sphere_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	sphere_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
	sphere_input.customPrimitiveArray.numPrimitives =
		static_cast<uint32_t>(spheres.size());
	sphere_input.customPrimitiveArray.flags = input_flags;
	sphere_input.customPrimitiveArray.numSbtRecords = num_sbt_records;
	sphere_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices;
	sphere_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes =
		sizeof(uint32_t);
	sphere_input.customPrimitiveArray.sbtIndexOffsetStrideInBytes =
		sizeof(uint32_t);

	buildGAS(context, gas, sphere_input);

	return d_sphere;
}

void* buildTriangleGAS(OptixDeviceContext context, GeometryAccelData& gas,
	const std::vector<float3>& vertices,
	std::vector<uint3>& triangles,
	const std::vector<uint32_t>& sbt_indices) {
	CUdeviceptr d_sbt_indices;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_indices),
		sizeof(uint32_t) * sbt_indices.size()));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_sbt_indices), sbt_indices.data(),
		sizeof(uint32_t) * sbt_indices.size(), cudaMemcpyHostToDevice));

	const size_t vertices_size = sizeof(float3) * vertices.size();
	CUdeviceptr d_vertices = 0;
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(),
		vertices_size, cudaMemcpyHostToDevice));

	const size_t tri_size = sizeof(uint3) * triangles.size();
	CUdeviceptr d_triangles = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_triangles), tri_size));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_triangles),
		triangles.data(), tri_size, cudaMemcpyHostToDevice));

	void* d_mesh_data;
	MeshData mesh_data{ reinterpret_cast<float3*>(d_vertices),
					   reinterpret_cast<uint3*>(d_triangles) };
	CUDA_CHECK(cudaMalloc(&d_mesh_data, sizeof(MeshData)));
	CUDA_CHECK(cudaMemcpy(d_mesh_data, &mesh_data, sizeof(MeshData),
		cudaMemcpyHostToDevice));

	uint32_t num_sbt_records = getNumSbtRecords(sbt_indices);
	gas.num_sbt_records = num_sbt_records;

	uint32_t* input_flags = new uint32_t[num_sbt_records];
	for (uint32_t i = 0; i < num_sbt_records; i++)
		input_flags[i] = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

	OptixBuildInput mesh_input = {};
	mesh_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	mesh_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	mesh_input.triangleArray.vertexStrideInBytes = sizeof(float3);
	mesh_input.triangleArray.numVertices =
		static_cast<uint32_t>(vertices.size());
	mesh_input.triangleArray.vertexBuffers = &d_vertices;
	mesh_input.triangleArray.flags = input_flags;
	mesh_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
	mesh_input.triangleArray.indexStrideInBytes = sizeof(uint3);
	mesh_input.triangleArray.indexBuffer = d_triangles;
	mesh_input.triangleArray.numIndexTriplets =
		static_cast<uint32_t>(triangles.size());
	mesh_input.triangleArray.numSbtRecords = num_sbt_records;
	mesh_input.triangleArray.sbtIndexOffsetBuffer = d_sbt_indices;
	mesh_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
	mesh_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

	buildGAS(context, gas, mesh_input);

	return d_mesh_data;
}

void buildIAS(OptixDeviceContext context, InstanceAccelData& ias,
	const std::vector<OptixInstance>& instances) {
	CUdeviceptr d_instances;
	const size_t instances_size = sizeof(OptixInstance) * instances.size();
	CUDA_CHECK(
		cudaMalloc(reinterpret_cast<void**>(&d_instances), instances_size));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_instances),
		instances.data(), instances_size,
		cudaMemcpyHostToDevice));

	OptixBuildInput instance_input = {};
	instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
	instance_input.instanceArray.instances = d_instances;
	instance_input.instanceArray.numInstances =
		static_cast<uint32_t>(instances.size());

	OptixAccelBuildOptions accel_options = {};
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
		OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;

	OptixAccelBufferSizes ias_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options,
		&instance_input,
		1,  // num build input
		&ias_buffer_sizes));

	size_t d_temp_buffer_size = ias_buffer_sizes.tempSizeInBytes;

	CUdeviceptr d_temp_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer),
		d_temp_buffer_size));

	CUdeviceptr d_buffer_temp_output_ias_and_compacted_size;
	size_t compacted_size_offset =
		roundUp<size_t>(ias_buffer_sizes.outputSizeInBytes, 8ull);
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&d_buffer_temp_output_ias_and_compacted_size),
		compacted_size_offset + 8));

	OptixAccelEmitDesc emit_property = {};
	emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_property.result =
		(CUdeviceptr)((char*)d_buffer_temp_output_ias_and_compacted_size +
			compacted_size_offset);

	OPTIX_CHECK(optixAccelBuild(context, 0, &accel_options, &instance_input,
		1,  // num build inputs
		d_temp_buffer, d_temp_buffer_size,
		// ias.d_output_buffer,
		d_buffer_temp_output_ias_and_compacted_size,
		ias_buffer_sizes.outputSizeInBytes,
		&ias.handle,  // emitted property list
		nullptr,      // num emitted property
		0));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));

	size_t compacted_ias_size;
	CUDA_CHECK(cudaMemcpy(&compacted_ias_size, (void*)emit_property.result,
		sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_ias_size < ias_buffer_sizes.outputSizeInBytes &&
		compacted_ias_size > 0) {
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ias.d_output_buffer),
			compacted_ias_size));
		OPTIX_CHECK(optixAccelCompact(context, 0, ias.handle,
			ias.d_output_buffer, compacted_ias_size,
			&ias.handle));
		CUDA_CHECK(
			cudaFree((void*)d_buffer_temp_output_ias_and_compacted_size));
	}
	else {
		ias.d_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
	}

	// if error compaction
	// ias.d_output_buffer = d_buffer_temp_output_ias_and_compacted_size;
}

OptixAabb read_obj_mesh(const std::string& obj_filename,
	std::vector<float3>& vertices,
	std::vector<uint3>& triangles) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warn;
	std::string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
		obj_filename.c_str());
	OptixAabb aabb;
	aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
	aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return aabb;
	}

	for (size_t s = 0; s < shapes.size(); s++) {
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			auto vertexOffset = vertices.size();

			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				if (idx.vertex_index >= 0) {
					tinyobj::real_t vx =
						attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t vy =
						attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t vz =
						attrib.vertices[3 * idx.vertex_index + 2];

					vertices.push_back(make_float3(vx, vy, vz));

					// Update aabb
					aabb.minX = std::min(aabb.minX, vx);
					aabb.minY = std::min(aabb.minY, vy);
					aabb.minZ = std::min(aabb.minZ, vz);

					aabb.maxX = std::max(aabb.maxX, vx);
					aabb.maxY = std::max(aabb.maxY, vy);
					aabb.maxZ = std::max(aabb.maxZ, vz);
				}
			}
			index_offset += fv;

			triangles.push_back(
				make_uint3(vertexOffset, vertexOffset + 1, vertexOffset + 2));
		}
	}
	return aabb;
}

enum class ShapeType { Mesh, Sphere };

#endif
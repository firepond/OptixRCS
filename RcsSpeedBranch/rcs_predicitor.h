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

#include <chrono>
#include <iostream>
#include <execution>


#include "RcsSpeedBranch/rcs_params.h"

#ifndef TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#endif

using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

#ifndef TRIANGLES_RCS_SPEED
#define TRIANGLES_RCS_SPEED

template <typename T>
struct SbtRecord {
	__align__(
		OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

enum class ShapeType { Mesh };

static void ContextLog(unsigned int level, const char* tag, const char* message,
	void* /*cbdata */) {
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag
		<< "]: " << message << "\n";
}

struct GeometryAccelData {
	OptixTraversableHandle handle;
	CUdeviceptr d_output_buffer;
	uint32_t num_sbt_records;
};

struct InstanceAccelData {
	OptixTraversableHandle handle;
	CUdeviceptr d_output_buffer;
	CUdeviceptr d_instances_buffer;
};

void BuildGAS(OptixDeviceContext context, GeometryAccelData& gas,
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
		&emit_property, 1));

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



uint32_t GetNumSbtRecords(const vector<uint32_t>& sbt_indices) {
	std::vector<uint32_t> sbt_counter;
	for (const uint32_t& sbt_idx : sbt_indices) {
		auto itr = std::find(sbt_counter.begin(), sbt_counter.end(), sbt_idx);
		if (sbt_counter.empty() || itr == sbt_counter.end())
			sbt_counter.emplace_back(sbt_idx);
	}
	return static_cast<uint32_t>(sbt_counter.size());
}



void* BuildTriangleGAS(OptixDeviceContext context, GeometryAccelData& gas,
	const vector<float3>& vertices, vector<uint3>& triangles,
	const vector<uint32_t>& sbt_indices) {
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

	uint32_t num_sbt_records = GetNumSbtRecords(sbt_indices);
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

	BuildGAS(context, gas, mesh_input);

	return d_mesh_data;
}



void BuildIAS(OptixDeviceContext context, InstanceAccelData& ias,
	const vector<OptixInstance>& instances) {
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
}



OptixAabb ReadObjMesh(const string& obj_filename, vector<float3>& vertices,
	vector<uint3>& triangles) {
	tinyobj::attrib_t attrib;
	vector<tinyobj::shape_t> shapes;
	vector<tinyobj::material_t> materials;

	string warn;
	string err;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
		obj_filename.c_str());
	OptixAabb aabb;
	aabb.minX = aabb.minY = aabb.minZ = std::numeric_limits<float>::max();
	aabb.maxX = aabb.maxY = aabb.maxZ = -std::numeric_limits<float>::max();

	if (!err.empty()) {
		std::cerr << err << std::endl;
		return aabb;
	}

	for (unsigned int s = 0; s < shapes.size(); s++) {
		unsigned int index_offset = 0;
		for (unsigned int f = 0; f < shapes[s].mesh.num_face_vertices.size();
			f++) {
			int fv = shapes[s].mesh.num_face_vertices[f];

			auto vertexOffset = vertices.size();

			for (unsigned int v = 0; v < fv; v++) {
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




class RcsPredictor {
private:
	const double c = 299792458.0;
	const uint32_t max_trace_depth = 10;

	OptixAabb aabb;
	char log[2048];  // For error reporting from OptiX creation functions
	vector<float3> vertices;
	vector<uint3> mesh_indices;

	float3 min_mesh;
	float3 max_mesh;
	float3 center;
	float3 polarization;

	double radius;
	double lamada;
	double lamda_nums;
	float waveNum;

	int rays_dimension;
	int rays_per_lamada;
	int size;

	float3* out_normals;
	float3* out_normal_device;

	Result* results;
	Result* device_ptr;

	Params params;

	OptixDeviceContext context = nullptr;
	OptixPipeline pipeline = nullptr;
	OptixModule module = nullptr;
	OptixProgramGroup raygen_prog_group = nullptr;
	OptixProgramGroup miss_prog_group = nullptr;
	OptixProgramGroup hitgroup_prog_group_triangle = nullptr;

	OptixPipelineCompileOptions pipeline_compile_options = {};
	OptixShaderBindingTable sbt = {};

	InstanceAccelData ias;
	std::vector<std::pair<ShapeType, HitGroupData>> hitgroup_datas;

	CUstream stream;

	void initOptix();
	void calculateOrientation();
	void calculateOutnormal();

public:
	bool is_debug = false;

	RcsPredictor();

	~RcsPredictor();

	void RcsPredictor::init(const string& obj_filename, int rays_per_lamada,
		double freq);

	double RcsPredictor::CalculateRcs(double phi, double theta);

};



RcsPredictor::RcsPredictor() {}



RcsPredictor::~RcsPredictor() {
	//
	// Cleanup
	//
	cout << "cleaning optix" << endl;
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_ptr)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(out_normal_device)));

	OPTIX_CHECK(optixPipelineDestroy(pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_triangle));
	OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));

	OPTIX_CHECK(optixModuleDestroy(module));

	OPTIX_CHECK(optixDeviceContextDestroy(context));

	free(results);
	free(out_normals);
}



void RcsPredictor::calculateOutnormal() {
	int triangle_num = mesh_indices.size();
	out_normals = (float3*)malloc(sizeof(float3) * triangle_num);

	for (int i = 0; i < triangle_num; i++) {
		uint3 index = mesh_indices[i];

		float3 v0 = vertices[index.x];
		float3 v1 = vertices[index.y];
		float3 v2 = vertices[index.z];

		float3 out_normal = normalize(cross((v1 - v0), (v2 - v0)));
		out_normals[i] = out_normal;
	}

	// allocate gpu memory to gpu pointer
	CUDA_CHECK(cudaMalloc((void**)&out_normal_device, sizeof(float3) * triangle_num));
	// copy data from host to device
	CUDA_CHECK(cudaMemcpy(out_normal_device, out_normals, sizeof(float3) * triangle_num,
		cudaMemcpyHostToDevice));
	CUDA_SYNC_CHECK();
	params.out_normals = out_normal_device;
}

void RcsPredictor::calculateOrientation() {
	float2 outDirSph = params.observer_pos;
	float3 dirN = make_float3(0);  // ray direction
	float3 dirU = make_float3(0);
	float3 dirR = make_float3(0);

	float cp = cosf(outDirSph.x);
	float sp = sinf(outDirSph.x);
	float ct = cosf(outDirSph.y);
	float st = sinf(outDirSph.y);

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

	float3 boundBoxCenter = center;
	float boundBoxRadius = radius;
	float3 rayPoolCenter = boundBoxCenter + dirN * 2.0 * boundBoxRadius;
	float3 rayPoolRectMin = rayPoolCenter - (dirR + dirU) * boundBoxRadius;
	float3 rayPoolRectMax = rayPoolCenter + (dirR + dirU) * boundBoxRadius;

	int rayCountSqrt = rays_dimension;

	float rayTubeRadius = boundBoxRadius / rayCountSqrt;
	float rayTubeDiameter = rayTubeRadius * 2.0f;
	float3 rayPosStepU = rayTubeDiameter * dirU;
	float3 rayPosStepR = rayTubeDiameter * dirR;
	float3 rayPosBegin = rayPoolRectMin + (rayPosStepU + rayPosStepR) / 2.0f;
	params.rayPosBegin = rayPosBegin;
	params.rayPosStepR = rayPosStepR;
	params.rayPosStepU = rayPosStepU;
	params.rayDir = -dirN;

	if (params.type == HH) {
		polarization = make_float3(-sp, cp, 0.0f);
	}
	else if (params.type == VV) {
		polarization = make_float3(cp * ct, sp * ct, -st);
	}
	else {
		polarization = make_float3(-sp, cp, 0.0f);
	}
	params.polarization = polarization;

}

void RcsPredictor::init(const string& obj_filename, int rays_per_lamada,
	double freq) {
	this->rays_per_lamada = rays_per_lamada;
	float waveLen = c / freq;
	float waveNum = 2 * M_PIf / waveLen;

	params.waveNum = waveNum;

	aabb = ReadObjMesh(obj_filename, vertices, mesh_indices);

	min_mesh = make_float3(aabb.minX, aabb.minY, aabb.minZ);
	max_mesh = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

	center = (min_mesh + max_mesh) / 2;
	radius = length(min_mesh - max_mesh) / 2.0f;
	lamada = c / freq;
	lamda_nums = radius * 2 / lamada;
	rays_dimension = ceil(lamda_nums * rays_per_lamada + 1.0f) + 1;
	size = rays_dimension * rays_dimension;
	if (is_debug) {
		cout << "using " << rays_dimension << " rays perdimension" << endl;
	}

	float rayRadius = radius / rays_dimension;
	float rayDiameter = rayRadius * 2;

	float rayArea = rayDiameter * rayDiameter;

	float t_value = (waveNum * rayArea) / (4.0f * M_PIf);
	params.t_value = t_value;
	initOptix();
	calculateOrientation();
	calculateOutnormal();

}

void RcsPredictor::initOptix() {
	// Initialize CUDA and create OptiX context

	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	// Initialize the OptiX API, loading all API entry points
	OPTIX_CHECK(optixInit());

	// Specify context options
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &ContextLog;
	if (is_debug) {
		options.logCallbackLevel = 4;
	}
	else {
		options.logCallbackLevel = 0;
	}

	// Associate a CUDA context (and therefore a specific GPU) with this
	// device context
	CUcontext cu_context = 0;  // zero means take the current context
	OPTIX_CHECK(optixDeviceContextCreate(cu_context, &options, &context));

	// Use default options for simplicity.  In a real use case we would
	// want to enable compaction, etc
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

	// Triangle build input

	std::vector<uint32_t> mesh_sbt_indices;

	// set sbt index (only one material available)
	const uint32_t sbt_index = 0;
	for (size_t i = 0; i < mesh_indices.size(); i++) {
		mesh_sbt_indices.push_back(sbt_index);
	}

	// accel handling
	// build mesh GAS
	GeometryAccelData mesh_gas;
	void* d_mesh_data;
	d_mesh_data = BuildTriangleGAS(context, mesh_gas, vertices, mesh_indices,
		mesh_sbt_indices);

	// HitGroupData
	hitgroup_datas.emplace_back(ShapeType::Mesh, HitGroupData{ d_mesh_data });

	std::vector<OptixInstance> instances;
	uint32_t flags = OPTIX_INSTANCE_FLAG_NONE;

	uint32_t sbt_offset = 0;
	uint32_t instance_id = 0;
	instances.emplace_back(OptixInstance{ {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
										 instance_id,
										 sbt_offset,
										 255,
										 flags,
										 mesh_gas.handle,
										 {0, 0} });

	BuildIAS(context, ias, instances);

	//
	// Create module
	//

	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount =
		OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

	pipeline_compile_options.usesMotionBlur = false;
	pipeline_compile_options.traversableGraphFlags =
		OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
	pipeline_compile_options.numPayloadValues = 2;
	pipeline_compile_options.numAttributeValues = 5;

	pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;

	pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
	pipeline_compile_options.usesPrimitiveTypeFlags =
		OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

	size_t inputSize = 0;
	size_t sizeof_log = sizeof(log);
	std::cout << OPTIX_SAMPLE_DIR << std::endl;
	const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR,
		"triangles_rcs.cu", inputSize);

	OPTIX_CHECK_LOG(optixModuleCreate(context, &module_compile_options,
		&pipeline_compile_options, input,
		inputSize, log, &sizeof_log, &module));

	//
	// Create program groups
	//

	{
		// Initialize to zeros
		OptixProgramGroupOptions program_group_options = {};

		OptixProgramGroupDesc raygen_prog_group_desc = {};  //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context, &raygen_prog_group_desc,
			1,  // num program groups
			&program_group_options, log, &sizeof_log, &raygen_prog_group));

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &miss_prog_group_desc,
			1,  // num program groups
			&program_group_options, log,
			&sizeof_log, &miss_prog_group));

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH =
			"__closesthit__triangle";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(
			optixProgramGroupCreate(context, &hitgroup_prog_group_desc,
				1,  // num program groups
				&program_group_options, log, &sizeof_log,
				&hitgroup_prog_group_triangle));
	}

	//
	// Link pipeline
	//
	{
		OptixProgramGroup program_groups[] = {
			raygen_prog_group, miss_prog_group, hitgroup_prog_group_triangle };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = max_trace_depth;

		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixPipelineCreate(
			context, &pipeline_compile_options, &pipeline_link_options,
			program_groups, sizeof(program_groups) / sizeof(program_groups[0]),
			log, &sizeof_log, &pipeline));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : program_groups) {
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes,
				pipeline));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(
			&stack_sizes, max_trace_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(
			pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			2  // maxTraversableDepth
		));
	}

	//
	// Set up shader binding table
	//
	{
		CUdeviceptr raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record),
			raygen_record_size));
		RayGenSbtRecord rg_sbt;

		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt,
			raygen_record_size, cudaMemcpyHostToDevice));

		CUdeviceptr miss_record;
		size_t miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record),
			miss_record_size));
		MissSbtRecord ms_sbt;
		ms_sbt.data = { 0.0f, 0.0f, 0.0f };
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt,
			miss_record_size, cudaMemcpyHostToDevice));

		// HitGroup
		CUdeviceptr hitgroup_record;
		size_t hitgroup_record_size =
			sizeof(HitGroupSbtRecord) * hitgroup_datas.size();  // triangle
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record),
			hitgroup_record_size));

		int hit_idx = 0;
		HitGroupSbtRecord* hg_sbt =
			new HitGroupSbtRecord[hitgroup_datas.size()];
		HitGroupData data = hitgroup_datas[hit_idx].second;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_triangle,
			&hg_sbt[hit_idx]));
		hg_sbt[hit_idx].data = data;

		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void**>(hitgroup_record), hg_sbt,
			hitgroup_record_size, cudaMemcpyHostToDevice));

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = hitgroup_record;
		sbt.hitgroupRecordStrideInBytes =
			static_cast<uint32_t>(sizeof(HitGroupSbtRecord));
		sbt.hitgroupRecordCount = static_cast<uint32_t>(hitgroup_datas.size());
	}
	/*CUstream stream;*/
	CUDA_CHECK(cudaStreamCreate(&stream));

	results = (Result*)malloc(sizeof(Result) * size);

	// allocate gpu memory to gpu pointer
	CUDA_CHECK(cudaMalloc((void**)&device_ptr, sizeof(Result) * size));
	// copy data from host to device
	CUDA_CHECK(cudaMemcpy(device_ptr, results, sizeof(Result) * size,
		cudaMemcpyHostToDevice));
	CUDA_SYNC_CHECK();

	//params.rays_per_dimension = rays_dimension;
	params.handle = ias.handle;
	params.type = VV;

	params.result = device_ptr;

}



double RcsPredictor::CalculateRcs(double phi, double theta) {
	// phi theta in radian
	float2 observer_pos = make_float2(phi, theta);

	//
	// launch
	//

	params.observer_pos = observer_pos;
	calculateOrientation();

	auto optix_start = high_resolution_clock::now();

	CUdeviceptr d_param;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_param), &params,
		sizeof(Params), cudaMemcpyHostToDevice));

	OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt,
		rays_dimension, rays_dimension, /*depth=*/1));
	CUDA_SYNC_CHECK();

	auto optix_end = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(optix_end - optix_start);
	std::cout << "optix time usage: " << ms_int.count() << "ms\n";

	auto copy_start = high_resolution_clock::now();
	CUDA_CHECK(cudaMemcpy(results, device_ptr, sizeof(Result) * size,
		cudaMemcpyDeviceToHost));
	CUDA_SYNC_CHECK();

	auto copy_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(copy_end - copy_start);

	std::cout << "result copy time usage: " << ms_int.count() << "ms\n";

	auto sum_start = high_resolution_clock::now();

	double au_real = 0;
	double au_img = 0;
	double ar_real = 0;
	double ar_img = 0;

#pragma omp parallel for reduction (+:au_real, au_img, ar_real, ar_img)
	for (int i = 0; i < size; i++) {
		Result cur_result = results[i];
		if (cur_result.refCount > 0) {
			au_real += cur_result.au_real;
			au_img += cur_result.au_img;
			ar_real += cur_result.ar_real;
			ar_img += cur_result.ar_img;
		}
	}
	std::complex<double> au = std::complex<double>(au_real, au_img);
	std::complex<double> ar = std::complex<double>(ar_real, ar_img);
	double ausq = pow(abs(au), 2);
	double arsq = pow(abs(ar), 2);
	double rcs_ori = 4.0 * M_PI * (ausq + arsq);  // * 4 * pi

	auto sum_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(sum_end - sum_start);
	if (is_debug) {
		std::cout << "rcs sum time usage: " << ms_int.count() << "ms\n";
		cout << "au : " << au << endl;
		cout << "ar : " << ar << endl;
	}

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
	return rcs_ori;
}

#endif
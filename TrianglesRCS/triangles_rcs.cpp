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

#include "triangles_rcs.h"
#include "TrianglesRCS/reduce.cu"
#include "rcs_params.h"

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

int main(int argc, char* argv[]) {
	string testfile = "corner_reflector";
	// string testfile = "d_reflector";
	// string testfile = "pone_0253743_model_cuboid_and_semishpere";
	string obj_file =
		"C:/development/optix/OptixRCS/resources/" + testfile + ".obj";

	float phi = 45;
	float theta = 60;
	int rays_per_dimension = 3000;
	if (argc > 1) {
		phi = std::atof(argv[1]);
		theta = std::atof(argv[2]);
		rays_per_dimension = std::stoi(argv[3]);
	}

	std::ofstream outtext;
	outtext.open("C:/development/optix/OptixRCS/output/rcs.csv", std::ios::app);

	OptixAabb aabb;
	std::vector<float3> vertices;
	std::vector<uint3> mesh_indices;

	aabb = read_obj_mesh(obj_file, vertices, mesh_indices);

	float3 minMesh =
		make_float3(aabb.minX * 1.5, aabb.minY * 1.5, aabb.minZ * 1.5);

	float3 maxMesh = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);
	float3 center = (minMesh + maxMesh) / 2;
	float3 zero_point = make_float3(0.0, 0.0, 0.0);
	float distance =
		sqrt(pow((minMesh.x - maxMesh.x), 2) + pow((minMesh.y - maxMesh.y), 2) +
			pow((minMesh.z - maxMesh.z), 2));

	float radius = distance / 2.0f;


	float thetaRadian = theta * M_PIf / 180.0f;  // radian of elevation

	int num_sphere = 1;  // 101

	float phiRadian = phi * M_PIf / 180.0f;  // radian of phi

	float3 cam_pos = make_float3(radius, phiRadian, thetaRadian);
	double rcs_ori = calculateRcs(vertices, mesh_indices, cam_pos, rays_per_dimension, center);
	double rcs = 10 * log10(rcs_ori);
	// phi and rcs
	cout << "rcs: " << rcs << endl;
	cout << "rcs ori : " << rcs_ori << endl;

	cout << phi << ", " << theta << ", " << rays_per_dimension << ", " << rcs << endl;
	outtext << phi << ", " << theta << ", " << rays_per_dimension << ", " << rcs << ", " << endl;

	outtext.close();

	return 0;
}

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

	double phi = 45;
	double theta = 60;
	double freq = 3E9;
	string test_model = "corner_reflector";

	if (argc > 1) {
		phi = std::atof(argv[1]);
		theta = std::atof(argv[2]);
		freq = std::atof(argv[3]);
		test_model = string(argv[4]);
	}

	string rootPathPrefix = "C:/development/optix/OptixRCS";
	string obj_file = rootPathPrefix + "/resources/" + test_model + ".obj";
	string csv_file = rootPathPrefix + "/output/" + test_model + "_rcs.csv";

	double c = 299792458.0;
	double lambda = c / freq;
	int rays_per_lamada = 100;
	int rays_per_dimension = 3000;
	std::ofstream out_stream;
	out_stream.open(csv_file, std::ios::app);

	OptixAabb aabb;
	vector<float3> vertices;
	vector<uint3> mesh_indices;

	aabb = ReadObjMesh(obj_file, vertices, mesh_indices);

	float3 min_mesh =
		make_float3(aabb.minX, aabb.minY, aabb.minZ);
	float3 max_mesh = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

	float3 center = (min_mesh + max_mesh) / 2;

	double radius = length(min_mesh - max_mesh) / 2.0f;

	double theta_radian = theta * M_PIf / 180.0f;  // radian of elevation

	double phi_radian = phi * M_PIf / 180.0f;  // radian of phi
	
	float3 observer_pos = make_float3(radius, phi_radian, theta_radian);

	double rcs_ori = CalculateRcs(vertices, mesh_indices, observer_pos, rays_per_dimension, center, freq);

	double rcs = 10 * log10(rcs_ori);

	// phi and rcs
	cout << test_model << ": ";
	cout << "freq = " << freq << ", ";
	cout << "phi = " << phi << ", ";
	cout << "theta = " << theta << ", ";
	cout << "rcs_dbsm = " << rcs << ", ";
	cout << "rcs_sm = " << rcs_ori << endl;
	out_stream << freq << ", " << phi << ", " << theta << ", " << rcs << ", " << endl;

	out_stream.close();

	return 0;
}

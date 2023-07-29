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
	auto sum_start = high_resolution_clock::now();

	string test_model = "corner_reflector";

	string rootPathPrefix = "C:/development/optix/OptixRCS";

	double c = 299792458.0;
	int rays_per_dimension = 3000;
	// 3Ghz
	double freq = 3E9;

	// start and end included
	double phi_start = 0;
	double phi_end = 90;
	double phi_interval = 1;

	double theta_start = 60;
	double theta_end = 60;
	double theta_interval = 1;

	if (argc > 1) {
		// list structure: numpy style [start:end:step]
		freq = atof(argv[1]);

		string phi_str = string(argv[2]);
		vector<double> phi_result;
		std::stringstream phi_ss(phi_str);
		string token;

		while (std::getline(phi_ss, token, ',')) {
			phi_result.push_back(std::stod(token));
		}
		phi_start = phi_result[0];
		phi_end = phi_result[1];
		phi_interval = phi_result[2];

		string theta_str = string(argv[3]);
		vector<double> theta_result;
		std::stringstream theta_ss(theta_str);

		while (std::getline(theta_ss, token, ',')) {
			theta_result.push_back(std::stod(token));
		}
		theta_start = theta_result[0];
		theta_end = theta_result[1];
		theta_interval = theta_result[2];

		test_model = string(argv[4]);
	}

	string obj_file = rootPathPrefix + "/resources/" + test_model + ".obj";
	string csv_file = rootPathPrefix + "/output/" + test_model + "_rcs.csv";

	std::ofstream out_stream;
	out_stream.open(csv_file);

	OptixAabb aabb;
	vector<float3> vertices;
	vector<uint3> mesh_indices;

	aabb = ReadObjMesh(obj_file, vertices, mesh_indices);

	float3 min_mesh =
		make_float3(aabb.minX, aabb.minY, aabb.minZ);
	float3 max_mesh = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

	float3 center = (min_mesh + max_mesh) / 2;
	double radius = length(min_mesh - max_mesh) / 2.0f;
	cout << "model: " << test_model << endl;
	cout << "Radius: " << radius << endl;


	int phi_count = (int)(phi_end - phi_start) / phi_interval + 1;
	int theta_count = (int)(theta_end - theta_start) / theta_interval + 1;

	int rays_per_lamada = 100;
	cout << "Phi: [" << phi_start << ":" << phi_end << ":" << phi_count << "]" << endl;
	cout << "Theta: [" << theta_start << ":" << theta_end << ":" << theta_count << "]" << endl;


	// [0, (phi_count-1)]
	for (int phi_i = 0; phi_i < phi_count; phi_i++) {
		double cur_phi = phi_start + phi_interval * phi_i;
		for (int theta_i = 0; theta_i < theta_count; theta_i++) {
			double cur_theta = theta_start + theta_interval * theta_i;


			double theta_radian = cur_theta * M_PIf / 180.0f;  // radian of elevation
			double phi_radian = cur_phi * M_PIf / 180.0f;  // radian of phi

			float3 observer_pos = make_float3(radius, phi_radian, theta_radian);

			double rcs_ori = CalculateRcs(vertices, mesh_indices, observer_pos, rays_per_lamada, center, freq);
			//double rcs_ori = 100.0f;

			double rcs = 10 * log10(rcs_ori);

			// output format: freq, phi, theta, rcs
			cout << test_model << ": ";
			cout << "freq = " << freq << ", ";
			cout << "phi = " << cur_phi << ", ";
			cout << "theta = " << cur_theta << ", ";
			cout << "rcs_dbsm = " << rcs << ", ";
			cout << "rcs_sm = " << rcs_ori << endl << endl;
			out_stream << freq << ", " << cur_phi << ", " << cur_theta << ", " << rcs << ", " << endl;

		}
	}


	auto sum_end = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(sum_end - sum_start);
	std::cout << "rcs sum time usage for " << phi_count * theta_count << " points : " << ms_int.count() << "ms\n";

	out_stream.close();

	return 0;
}


#include <sampleConfig.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include "reduceKernels.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "RcsSpeedBranch/rcs_params.h"
#include "RcsSpeedBranch/rcs_predicitor.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

//using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

int main(int argc, char* argv[]) {
	auto program_start = high_resolution_clock::now();
	bool is_debug = false;
	string test_model = "a380";
	//string test_model = "corner_reflector";
	// string test_model = "large_trihedral_reflector";

	string rootPathPrefix = "C:/development/optix/OptixRCS";

	// double c = 299792458.0;
	// int rays_per_dimension = 3000;
	//  max dimension for RTX3060 6GB: 20000
	double freq = 1E9;
	int rays_per_lamada = 50;

	// start and end included
	double phi_start = 40;
	double phi_end = 50;
	double phi_interval = 5;

	double theta_start = 90;
	double theta_end = 90;
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

		rays_per_lamada = atoi(argv[4]);

		test_model = string(argv[5]);
	}

	string obj_file = rootPathPrefix + "/resources/" + test_model + ".obj";
	string csv_file = rootPathPrefix + "/output/" + test_model + "_rcs.csv";

	std::ofstream out_stream;
	out_stream.open(csv_file);

	int phi_count = ceil((phi_end - phi_start) / phi_interval) + 1;
	int theta_count = ceil((theta_end - theta_start) / theta_interval) + 1;

	cout << "Phi: [" << phi_start << ":" << phi_end << ":" << phi_count << "]"
		<< endl;
	cout << "Theta: [" << theta_start << ":" << theta_end << ":" << theta_count
		<< "]" << endl;

	auto init_start = high_resolution_clock::now();

	RcsPredictor predicitor;
	predicitor.is_debug = true;
	predicitor.centerRelocate = false;
	predicitor.init(obj_file, rays_per_lamada, freq);

	auto init_end = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(init_end - init_start);
	std::cout << "Environments init time:" << ms_int.count() << "ms\n";

	auto rcs_start = high_resolution_clock::now();
	// [0, (phi_count-1)]
	for (int phi_i = 0; phi_i < phi_count; phi_i++) {
		double cur_phi = phi_start + phi_interval * phi_i;
		// radian of phi
		double phi_radian = cur_phi * M_PI / 180.0;

		for (int theta_i = 0; theta_i < theta_count; theta_i++) {
			double cur_theta = theta_start + theta_interval * theta_i;

			// radian of elevation
			double theta_radian = cur_theta * M_PI / 180.0;

			double rcs_ori = predicitor.CalculateRcs(phi_radian, theta_radian);
			// double rcs_ori = 100.0f;

			double rcs = 10 * log10(rcs_ori);

			// output format: freq, phi, theta, rcs
			cout << test_model << ": ";
			cout << "freq = " << freq << ", ";
			cout << "phi = " << cur_phi << ", ";
			cout << "theta = " << cur_theta << ", ";
			cout << "rcs_dbsm = " << rcs << ", ";
			cout << "rcs_sm = " << rcs_ori << endl << endl;
			out_stream << freq << ", " << cur_phi << ", " << cur_theta << ", "
				<< rcs << ", " << endl;
		}
	}
	auto rcs_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(rcs_end - rcs_start);
	std::cout << "OptiX calculation time for " << phi_count * theta_count
		<< " points : " << ms_int.count() << "ms, averge: " << (double)ms_int.count() / phi_count * theta_count << endl;

	auto program_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(program_end - program_start);
	std::cout << "Whole program time for " << phi_count * theta_count
		<< " points : " << ms_int.count() << "ms, averge: " << (double)ms_int.count() / phi_count * theta_count << endl;

	out_stream.close();

	return 0;
}

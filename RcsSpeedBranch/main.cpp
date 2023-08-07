
#include <sampleConfig.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "RcsSpeedBranch/rcs_params.h"
#include "RcsSpeedBranch/rcs_predicitor.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// using std::complex;
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using json = nlohmann::json;

template <typename T>
void setConfigFromJson(T& variable, string variable_name, json data) {
	if (data.contains(variable_name)) {
		variable = data[variable_name];
	}
	else {
		cout << "Unspecified " << variable_name
			<< " option, using default: " << variable << endl;
	}
}

int main(int argc, char* argv[]) {
	auto program_start = high_resolution_clock::now();
	string config_file = "C:/development/optix/OptixRCS/test/default_config.json";
	if (argc > 1) {
		config_file = string(argv[1]);
	}
	else {
		cout << "Unspecified "
			<< " config file, using default: " << config_file << endl;
	}

	bool is_debug = false;
	bool center_relocate = false;

	int rays_per_wavelength = 10;
	int trace_depth = 10;

	//  max dimension for RTX3060 6GB: 20000
	double freq = 3E9;
	double reflectance = 1.0;

	// start and end included
	double phi_start = 0;
	double phi_end = 90;
	double phi_interval = 5;
	double theta_start = 90;
	double theta_end = 90;
	double theta_interval = 1;

	string model_file =
		"C:/development/optix/OptixRCS/resources/corner_reflector.obj";
	string csv_file =
		"C:/development/optix/OptixRCS/output/corner_reflector.csv";

	// read json configs
	std::ifstream f(config_file);
	json data = json::parse(f);
	setConfigFromJson(is_debug, "is_debug", data);
	setConfigFromJson(center_relocate, "center_relocate", data);
	setConfigFromJson(rays_per_wavelength, "rays_per_wavelength", data);
	setConfigFromJson(trace_depth, "trace_depth", data);
	setConfigFromJson(freq, "freq", data);
	setConfigFromJson(reflectance, "reflectance", data);
	setConfigFromJson(phi_start, "phi_start", data);
	setConfigFromJson(phi_end, "phi_end", data);
	setConfigFromJson(phi_interval, "phi_interval", data);
	setConfigFromJson(theta_start, "theta_start", data);
	setConfigFromJson(theta_end, "theta_end", data);
	setConfigFromJson(theta_interval, "theta_interval", data);
	setConfigFromJson(model_file, "model_file", data);
	setConfigFromJson(csv_file, "csv_file", data);
	PolarizationTypes polarization = HH;
	if (data.contains("polarization")) {
		string pol = data["polarization"];
		if (pol == "HH") {
			polarization = HH;
		}
		else if (pol == "VV") {
			polarization = VV;
		}
	}
	else {
		cout << "Unspecified polarization  option, using default: HH" << endl;
		polarization = HH;
	}
	size_t lastSlash = model_file.find_last_of("/");
	std::string model_name;
	if (lastSlash != std::string::npos) {
		model_name = model_file.substr(lastSlash + 1);  // +1 to get the characters after the slash
	}
	else {
		model_name = model_file;
	}

	std::ofstream out_stream;
	out_stream.open(csv_file);

	int phi_count = ceil((phi_end - phi_start) / phi_interval) + 1;
	int theta_count = ceil((theta_end - theta_start) / theta_interval) + 1;

	cout << "Phi: [" << phi_start << ":" << phi_end << ":" << phi_interval << "]"
		<< endl;
	cout << "Theta: [" << theta_start << ":" << theta_end << ":" << theta_interval
		<< "]" << endl;

	auto init_start = high_resolution_clock::now();

	RcsPredictor predicitor;
	predicitor.is_debug = is_debug;
	predicitor.centerRelocate = center_relocate;
	predicitor.max_trace_depth = trace_depth;
	predicitor.reflectance = reflectance;
	predicitor.type = polarization;
	predicitor.init(model_file, rays_per_wavelength, freq);

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

			double rcs = 10 * log10(rcs_ori);

			// output format: freq, phi, theta, rcs
			cout << model_name << ": ";
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
		<< " points : " << ms_int.count() << "ms, averge: "
		<< (double)ms_int.count() / phi_count * theta_count << endl;

	auto program_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(program_end - program_start);
	std::cout << "Whole program time for " << phi_count * theta_count
		<< " points : " << ms_int.count() << "ms, averge: "
		<< (double)ms_int.count() / phi_count * theta_count << endl;

	out_stream.close();

	return 0;
}

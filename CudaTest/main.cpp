#include <string>
#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

#include <sutil/Exception.h>


#include "reduceKernels.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using std::cout;
using std::endl;
using std::string;
int main(int argc, char* argv[]) {
	int size = 19068*19068*4;
	float* source_data = new float[size];

	memset(source_data, 2.0f, sizeof(float) * size);

	float* results_device;
	CUDA_CHECK(cudaMalloc((void**)&results_device, sizeof(float) * size));
	CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(results_device), source_data,
		sizeof(float) * size, cudaMemcpyHostToDevice));
	CUDA_SYNC_CHECK();

	int element_size = size / 4;

	cout << "Sum start: " << endl;
	auto sum_start = high_resolution_clock::now();
	float* result = reduce0(results_device, element_size);

	auto sum_end = high_resolution_clock::now();
	auto ms_int = duration_cast<milliseconds>(sum_end - sum_start);
	std::cout << result[0] << "Kernel 0 sum time:" << ms_int.count() << "ms\n";

	cout << "Sum start: " << endl;
	sum_start = high_resolution_clock::now();
	result = reduce1(results_device, element_size);

	sum_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(sum_end - sum_start);
	std::cout << result[0] << "Kernel 1 sum time:" << ms_int.count() << "ms\n";

	cout << "Sum start: " << endl;
	sum_start = high_resolution_clock::now();
	result = reduce2(results_device, size);

	sum_end = high_resolution_clock::now();
	ms_int = duration_cast<milliseconds>(sum_end - sum_start);
	std::cout << result[0] << "Kernel 2 sum time:" << ms_int.count() << "ms\n";



	return 0;

}

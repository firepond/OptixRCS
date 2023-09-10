#include <cuda_runtime.h>
#include <sutil/vec_math.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

#include "rcs_params.h"
#include "reduceKernels.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using std::cout;
using std::endl;
using std::string;

__device__ void warpReduce(volatile float* sdata, int tid) {
	sdata[4 * tid] += sdata[4 * (tid + 32)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 32) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 32) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 32) + 3];

	sdata[4 * tid] += sdata[4 * (tid + 16)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 16) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 16) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 16) + 3];

	sdata[4 * tid] += sdata[4 * (tid + 8)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 8) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 8) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 8) + 3];

	sdata[4 * tid] += sdata[4 * (tid + 4)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 4) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 4) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 4) + 3];

	sdata[4 * tid] += sdata[4 * (tid + 2)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 2) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 2) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 2) + 3];

	sdata[4 * tid] += sdata[4 * (tid + 1)];
	sdata[4 * tid + 1] += sdata[4 * (tid + 1) + 1];
	sdata[4 * tid + 2] += sdata[4 * (tid + 1) + 2];
	sdata[4 * tid + 3] += sdata[4 * (tid + 1) + 3];
}

__global__ void reduceKernel(float* g_idata, float* g_odata, int size) {
	extern __shared__ float sdata[512 * 4];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (i + blockDim.x < size) {
		sdata[tid * 4] = g_idata[i * 4] + g_idata[(i + blockDim.x) * 4];
		sdata[tid * 4 + 1] = g_idata[i * 4 + 1] + g_idata[(i + blockDim.x) * 4 + 1];
		sdata[tid * 4 + 2] = g_idata[i * 4 + 2] + g_idata[(i + blockDim.x) * 4 + 2];
		sdata[tid * 4 + 3] = g_idata[i * 4 + 3] + g_idata[(i + blockDim.x) * 4 + 3];
	}
	else {
		sdata[tid * 4] = g_idata[i * 4];
		sdata[tid * 4 + 1] = g_idata[i * 4 + 1];
		sdata[tid * 4 + 2] = g_idata[i * 4 + 2];
		sdata[tid * 4 + 3] = g_idata[i * 4 + 3];
	}

	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid * 4] = sdata[tid * 4] + sdata[(tid + s) * 4];
			sdata[tid * 4 + 1] = sdata[tid * 4 + 1] + sdata[(tid + s) * 4 + 1];
			sdata[tid * 4 + 2] = sdata[tid * 4 + 2] + sdata[(tid + s) * 4 + 2];
			sdata[tid * 4 + 3] = sdata[tid * 4 + 3] + sdata[(tid + s) * 4 + 3];
		}
		__syncthreads();
	}

	if (tid < 32) warpReduce(sdata, tid);

	// write result for this block to global mem
	unsigned int block_mem = blockIdx.x * 4;
	if (tid < 4) {
		g_odata[block_mem + tid] = sdata[tid];

	}
}
float* reduce(float* g_idata, int size) {
	int blockDim = 512;
	int reduceDim = blockDim * 2;
	int memoryDim = reduceDim * 4;
	int reduceCount = ceil(log2(size) / log2(reduceDim));

	float* out_device;
	float* to_reduce_device = g_idata;
	int block_count = ceil((double)size / reduceDim);
	cudaMalloc((void**)&out_device, sizeof(float) * block_count * 4);
	float* out_device_holder = out_device;

	while (size > 1) {
		if (size <= reduceDim) {
			reduceKernel << <1, blockDim >> > (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / reduceDim);
			std::cout << "k1 sum start: " << std::endl;
			auto sum_start = high_resolution_clock::now();
			reduceKernel << <block_count, blockDim >> > (to_reduce_device,
				out_device, size);
			cudaDeviceSynchronize();
			auto sum_end = high_resolution_clock::now();
			auto ms_int = duration_cast<std::chrono::microseconds>(sum_end - sum_start);
			std::cout << "kernel sum time:" << ms_int.count() << "us\n";
		}

		// swap to_reduce_device and out_device
		float* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	float* result_out = (float*)malloc(sizeof(float) * 4);
	cudaMemcpy(result_out, out_device, sizeof(float) * 4, cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	//cudaDeviceSynchronize();
	return result_out;
}

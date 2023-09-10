#include <cuda_runtime.h>
#include <sutil/vec_math.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

// #include <cuda_runtime.h>

// #include <sutil/Exception.h>

#include "reduceKernels.h"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

using std::cout;
using std::endl;
using std::string;

__global__ void reduceKernel0(float* g_idata, float* g_odata, int size) {
	extern __shared__ float sdata[512 * 4];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int tid_mem = tid * 4;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int i_mem = i * 4;
	if (i + blockDim.x < size) {
		sdata[tid_mem] = g_idata[i_mem];
		sdata[tid_mem + 1] = g_idata[i_mem + 1];
		sdata[tid_mem + 2] = g_idata[i_mem + 2];
		sdata[tid_mem + 3] = g_idata[i_mem + 3];

	}
	else {
		sdata[tid_mem] = 0;
		sdata[tid_mem + 1] = 0;
		sdata[tid_mem + 2] = 0;
		sdata[tid_mem + 3] = 0;
	}

	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			unsigned int s_id = 4 * (tid + s);
			sdata[tid_mem] += sdata[s_id];
			sdata[tid_mem + 1] += sdata[s_id + 1];
			sdata[tid_mem + 2] += sdata[s_id + 2];
			sdata[tid_mem + 3] += sdata[s_id + 3];
		}
		__syncthreads();
	}
	// write result for this block to global mem

	if (tid == 0) {
		unsigned int block_mem = blockIdx.x * 4;
		g_odata[block_mem] = sdata[0];
		g_odata[block_mem + 1] = sdata[1];
		g_odata[block_mem + 2] = sdata[2];
		g_odata[block_mem + 3] = sdata[3];
	}
}

__global__ void reduceKernel1(float* g_idata, float* g_odata, int size) {
	extern __shared__ float sdata[512 * 4];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

	if (i + blockDim.x < size) {
		sdata[tid * 4] = g_idata[i * 4] + g_idata[(i + blockDim.x) * 4];
		sdata[tid * 4 + 1] =
			g_idata[i * 4 + 1] + g_idata[(i + blockDim.x) * 4 + 1];
		sdata[tid * 4 + 2] =
			g_idata[i * 4 + 2] + g_idata[(i + blockDim.x) * 4 + 2];
		sdata[tid * 4 + 3] =
			g_idata[i * 4 + 3] + g_idata[(i + blockDim.x) * 4 + 3];
	}
	else {
		sdata[tid * 4] = g_idata[i * 4];
		sdata[tid * 4 + 1] = g_idata[i * 4 + 1];
		sdata[tid * 4 + 2] = g_idata[i * 4 + 2];
		sdata[tid * 4 + 3] = g_idata[i * 4 + 3];
	}

	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid * 4] = sdata[tid * 4] + sdata[(tid + s) * 4];
			sdata[tid * 4 + 1] = sdata[tid * 4 + 1] + sdata[(tid + s) * 4 + 1];
			sdata[tid * 4 + 2] = sdata[tid * 4 + 2] + sdata[(tid + s) * 4 + 2];
			sdata[tid * 4 + 3] = sdata[tid * 4 + 3] + sdata[(tid + s) * 4 + 3];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	unsigned int block_mem = blockIdx.x * 4;
	if (tid < 4) {
		g_odata[blockIdx.x * 4+tid] = sdata[tid];
	}
}

__global__ void reduceKernel2(float* g_idata, float* g_odata, int size) {
	extern __shared__ float sdata[512 * 4];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = 0.0f;
	sdata[tid + 1] = 0.0f;
	sdata[tid + 2] = 0.0f;
	sdata[tid + 3] = 0.0f;

	if (i < size) {
		sdata[tid * 4] = g_idata[i * 4] + g_idata[(i + blockDim.x) * 4];
		sdata[tid * 4 + 1] =
			g_idata[i * 4 + 1] + g_idata[(i + blockDim.x) * 4 + 1];
		sdata[tid * 4 + 2] =
			g_idata[i * 4 + 2] + g_idata[(i + blockDim.x) * 4 + 2];
		sdata[tid * 4 + 3] =
			g_idata[i * 4 + 3] + g_idata[(i + blockDim.x) * 4 + 3];
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

	if (tid < 32) {
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

	// write result for this block to global mem
	if (tid == 0) {
		g_odata[blockIdx.x * 4] = sdata[0];
		g_odata[blockIdx.x * 4 + 1] = sdata[1];
		g_odata[blockIdx.x * 4 + 2] = sdata[2];
		g_odata[blockIdx.x * 4 + 3] = sdata[3];
	}
}

__global__ void reduceKernelAllSum(float* g_idata, float* g_odata, int size) {
	extern __shared__ float sdata[512*4];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = 0.0f;

	if (i < size) {
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

	}

	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] = sdata[tid] + sdata[tid + s];

		}
		__syncthreads();
	}

	if (tid < 32) {
		sdata[tid] += sdata[tid + 32];


		sdata[tid] += sdata[tid + 16];

		sdata[tid] += sdata[tid + 8];

		sdata[tid] += sdata[tid + 4];


		sdata[tid] += sdata[tid + 2];

		sdata[tid] += sdata[tid + 1];

	}

	// write result for this block to global mem
	if (tid == 0) {
		g_odata[blockIdx.x * 4] = sdata[0];
		g_odata[blockIdx.x * 4 + 1] = sdata[1];
		g_odata[blockIdx.x * 4 + 2] = sdata[2];
		g_odata[blockIdx.x * 4 + 3] = sdata[3];
	}
}



float* reduce0(float* g_idata, int size) {
	int blockDim = 512;
	int reduceDim = blockDim;
	int memoryDim = reduceDim * 4;
	int reduceCount = ceil(log2(size) / log2(reduceDim));

	float* out_device;
	float* to_reduce_device = g_idata;
	int block_count = ceil((double)size / reduceDim);
	cudaMalloc((void**)&out_device, sizeof(float) * block_count * 4);
	float* out_device_holder = out_device;

	while (size > 1) {
		if (size <= reduceDim) {
			reduceKernel0 << <1, blockDim >> > (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / reduceDim);
			reduceKernel0 << <block_count, blockDim >> > (to_reduce_device,
				out_device, size);
		}

		// swap to_reduce_device and out_device
		float* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	float* result_out = (float*)malloc(sizeof(float) * 4);
	cudaMemcpy(result_out, out_device, sizeof(float) * 4,
		cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	// cudaDeviceSynchronize();
	return result_out;
}

float* reduce1(float* g_idata, int size) {
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
			reduceKernel1 << <1, blockDim >> > (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / reduceDim);
			std::cout << "k1 sum start: " << std::endl;
			auto sum_start = high_resolution_clock::now();
			reduceKernel1 << <block_count, blockDim >> > (to_reduce_device,
				out_device, size);
			cudaDeviceSynchronize();
			auto sum_end = high_resolution_clock::now();
			auto ms_int = duration_cast<milliseconds>(sum_end - sum_start);
			std::cout << "k1 sum time:" << ms_int.count() << "ms\n";
		}

		// swap to_reduce_device and out_device
		float* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	float* result_out = (float*)malloc(sizeof(float) * 4);
	cudaMemcpy(result_out, out_device, sizeof(float) * 4,
		cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	// cudaDeviceSynchronize();
	return result_out;
}

float* reduce2(float* g_idata, int size) {
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
			reduceKernelAllSum << <1, blockDim >> > (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / reduceDim);
			std::cout << "k2 sum start: " << std::endl;
			auto sum_start = high_resolution_clock::now();
			reduceKernelAllSum << <block_count, blockDim >> > (to_reduce_device,
				out_device, size);
			cudaDeviceSynchronize();
			auto sum_end = high_resolution_clock::now();
			auto ms_int = duration_cast<milliseconds>(sum_end - sum_start);
			std::cout << "k2 sum time:" << ms_int.count() << "ms\n";
		}

		// swap to_reduce_device and out_device
		float* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	float* result_out = (float*)malloc(sizeof(float) * 4);
	cudaMemcpy(result_out, out_device, sizeof(float) * 4,
		cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	// cudaDeviceSynchronize();
	return result_out;
}


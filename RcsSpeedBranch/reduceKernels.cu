#include <cuda_runtime.h>
#include <sutil/vec_math.h>

#include <cmath>

#include "rcs_params.h"
#include "reduceKernels.h"


__global__ void reduceKernel(Result* g_idata, Result* g_odata, int size) {
	extern __shared__ Result sdata[512 * 4];

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
	if (tid == 0) {
		g_odata[blockIdx.x * 4] = sdata[0];
		g_odata[blockIdx.x * 4 + 1] = sdata[1];
		g_odata[blockIdx.x * 4 + 2] = sdata[2];
		g_odata[blockIdx.x * 4 + 3] = sdata[3];
	}
}

Result* reduce(Result* g_idata, int size) {
	int blockDim = 512;
	int reduceDim = blockDim * 2;
	int memoryDim = reduceDim * 4;
	int reduceCount = ceil(log2(size) / log2(reduceDim));

	Result* out_device;
	Result* to_reduce_device = g_idata;
	int block_count = ceil((double)size / reduceDim);
	cudaMalloc((void**)&out_device, sizeof(Result) * block_count * 4);
	Result* out_device_holder = out_device;
	//cudaDeviceSynchronize();

	while (size > 1) {
		if (size <= reduceDim) {
			reduceKernel << <1, blockDim >> > (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / reduceDim);
			reduceKernel << <block_count, blockDim >> > (to_reduce_device,
				out_device, size);
		}

		//cudaDeviceSynchronize();

		// swap to_reduce_device and out_device
		Result* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	Result* result_out = (float*)malloc(sizeof(float) * 4);
	cudaMemcpy(result_out, out_device, sizeof(float) * 4, cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	//cudaDeviceSynchronize();
	return result_out;
}

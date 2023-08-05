//#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include "reduceKernels.h"
#include "rcs_params.h"
#include <sutil/vec_math.h>
#include <cmath>

__constant__ Result zero;


__global__ void reduceKernel(Result* g_idata, Result* g_odata, int size) {

	int temp = (blockIdx.x + 1) * blockDim.x;

	int curBlockSize = (temp <= size) * blockDim.x + (temp > size) * (size % blockDim.x);
	extern __shared__ Result sdata[512];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < curBlockSize) {
		sdata[tid] = g_idata[i];
	}
	else {
		sdata[tid] = zero;
	}

	__syncthreads();
	// do reduction in shared mem

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid].ar_img += sdata[tid + s].ar_img;
			sdata[tid].ar_real += sdata[tid + s].ar_real;
			sdata[tid].au_img += sdata[tid + s].au_img;
			sdata[tid].au_real += sdata[tid + s].au_real;
	
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
	}

}


Result reduce(Result* g_idata, int size)
{
	int blockDim = 512;

	int reduceCount = ceil(log2(size) / log2(blockDim));

	Result* out_device;
	Result* to_reduce_device = g_idata;
	int block_count = ceil((double)size / blockDim);
	cudaMalloc((void**)&out_device, sizeof(Result) * block_count);
	Result* out_device_holder = out_device;
	cudaDeviceSynchronize();

	while (size > 1) {
		if (size <= blockDim) {
			reduceKernel <<< 1, blockDim >>> (to_reduce_device, out_device, size);
			break;
		}
		else {
			block_count = ceil((double)size / blockDim);
			reduceKernel <<< block_count, blockDim >>> (to_reduce_device, out_device, size);
		}
	
		cudaDeviceSynchronize();

		// swap to_reduce_device and out_device
		Result* temp;
		temp = to_reduce_device;
		to_reduce_device = out_device;
		out_device = temp;

		size = block_count;
	}

	Result result_out;
	cudaMemcpy(&result_out, out_device, sizeof(Result),
		cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device_holder));
	cudaDeviceSynchronize();
	return result_out;
}


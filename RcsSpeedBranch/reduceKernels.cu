//#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include "reduceKernels.h"
#include "rcs_params.h"
#include <sutil/vec_math.h>
#include <cmath>

__global__ void reduceKernel(Result* g_idata, Result* g_odata, int size) {

	int temp = (blockIdx.x + 1) * blockDim.x;

	int curBlockSize = (temp <= size) * blockDim.x + (temp > size) * (size % blockDim.x);
	extern __shared__ Result sdata[512];
	Result zero;
	zero.ar_img = 0.0f;
	zero.ar_real = 0.0f;
	zero.au_img = 0.0f;
	zero.au_real = 0.0f;
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
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			//Result rl = sdata[tid];
			//Result rr = sdata[tid + s];
			sdata[tid].ar_img += sdata[tid + s].ar_img;
			sdata[tid].ar_real += sdata[tid + s].ar_real;
			sdata[tid].au_img += sdata[tid + s].au_img;
			sdata[tid].au_real += sdata[tid + s].au_real;
			sdata[tid].refCount += sdata[tid + s].refCount;

			//sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	//__syncthreads();
	// write result for this block to global mem
	if (tid == 0) {
		g_odata[blockIdx.x] = sdata[0];
		//g_odata[blockIdx.x].refCount = 2;
	}

}


Result reduce(Result* g_idata, int size)
{
	int blockDim = 512;

	int reduceCount = ceil(log2(size) / log2(blockDim));



	Result* out_device;
	Result* to_reduce_device = g_idata;
	int block_count = ceil(size / blockDim);
	while (size > 1) {
		block_count = ceil((double)size / blockDim);
		// allocate gpu memory to gpu pointer
		cudaMalloc((void**)&out_device, sizeof(Result) * block_count);
		//cudaMemcpy(result_out_device, result_out, sizeof(Result) * block_count,
		//	cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		reduceKernel <<< block_count, blockDim >> > (to_reduce_device, out_device, size);
		cudaFree(reinterpret_cast<void*>(to_reduce_device));
		to_reduce_device = out_device;
		size = block_count;
	}


	//Result result_out = (Result*)malloc(sizeof(Result) * block_count);
	Result result_out;
	cudaMemcpy(&result_out, out_device, sizeof(Result),
		cudaMemcpyDeviceToHost);
	cudaFree(reinterpret_cast<void*>(out_device));
	cudaDeviceSynchronize();
	return result_out;
}


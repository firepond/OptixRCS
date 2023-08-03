#pragma once

#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <cuda.h>
#include "RcsSpeedBranch/rcs_params.h"

__global__ void reduce(int* g_idata, int* g_odata) {

    __shared__ int sdata[256];

    // each thread loads one element from global to shared mem
    // note use of 1D thread indices (only) in this kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[threadIdx.x] = g_idata[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * threadIdx.x;;

        if (index < blockDim.x)
        {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (threadIdx.x == 0)
        atomicAdd(g_odata, sdata[0]);
}


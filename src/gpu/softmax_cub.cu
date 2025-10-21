#include "softmax.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in file " \
                  << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void fused_softmax_kernel(const float* input, float* output, size_t N, float* temp_storage) {
    cg::grid_group grid = cg::this_grid();
    constexpr int block_size = 1024;
    
    // Shared memory for manual max reduction
    __shared__ float s_max[block_size];

    // --- Phase 1: Find Global Maximum ---
    // Part 1: Block-level reduction for Max using manual shared memory reduction
    float thread_max = -INFINITY;
    for (size_t i = grid.thread_rank(); i < N; i += grid.size()) {
        thread_max = max(thread_max, input[i]);
    }

    s_max[threadIdx.x] = thread_max;
    __syncthreads();

    // Perform the reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + s]);
        }
        __syncthreads();
    }

    // The block's maximum value is now in s_max[0]
    if (threadIdx.x == 0) {
        temp_storage[blockIdx.x] = s_max[0];
    }
    
    grid.sync(); // Wait for all blocks to find their local max

    // Part 2: Grid-level reduction for Max (done by the first block)
    if (blockIdx.x == 0) {
        thread_max = -INFINITY;
        for (size_t i = threadIdx.x; i < grid.num_blocks(); i += blockDim.x) {
            thread_max = max(thread_max, temp_storage[i]);
        }
        
        s_max[threadIdx.x] = thread_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + s]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            temp_storage[0] = s_max[0]; // Final global max
        }
    }
    
    grid.sync();
    const float global_max = temp_storage[0];

    // --- Phase 2: Calculate Global Sum using CUB (this part works correctly) ---
    typedef cub::BlockReduce<float, block_size> BlockReduceSum;
    __shared__ typename BlockReduceSum::TempStorage reduce_smem_sum;

    float thread_sum = 0.0f;
    for (size_t i = grid.thread_rank(); i < N; i += grid.size()) {
        thread_sum += expf(input[i] - global_max);
    }
    float block_sum = BlockReduceSum(reduce_smem_sum).Sum(thread_sum);
    if (threadIdx.x == 0) temp_storage[blockIdx.x] = block_sum;
    
    grid.sync();

    if (blockIdx.x == 0) {
        thread_sum = 0.0f;
        for (size_t i = threadIdx.x; i < grid.num_blocks(); i += blockDim.x) {
            thread_sum += temp_storage[i];
        }
        block_sum = BlockReduceSum(reduce_smem_sum).Sum(thread_sum);
        if (threadIdx.x == 0) temp_storage[0] = block_sum;
    }
    
    grid.sync();
    const float global_sum = (temp_storage[0] > 0.0f) ? temp_storage[0] : 1.0f;

    // --- Phase 3: Final Normalization ---
    for (size_t i = grid.thread_rank(); i < N; i += grid.size()) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

void softmax_gpu_cub_fused(std::vector<float>& vec) {
    if (vec.empty()) return;
    size_t N = vec.size();
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, vec.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int threads_per_block = 1024;
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device_id));

    int num_blocks = properties.multiProcessorCount * 2;

    float* d_temp_storage;
    CUDA_CHECK(cudaMalloc(&d_temp_storage, num_blocks * sizeof(float)));

    void* kernel_args[] = { (void*)&d_input, (void*)&d_output, (void*)&N, (void*)&d_temp_storage };

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)fused_softmax_kernel, 
        num_blocks, 
        threads_per_block, 
        kernel_args
    ));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(vec.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);
}
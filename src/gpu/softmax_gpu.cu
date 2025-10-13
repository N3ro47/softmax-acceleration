#include <stdio.h>
#include <float.h>
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

static inline __device__ float atomicMax(float *addr, float value) {
    int *addr_as_int = (int *)addr;
    int old = *addr_as_int;
    int assumed;

    do {
        assumed = old;
        if (__int_as_float(assumed) >= value)
            break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void _sum_vec(float* vec, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
        vec[i] = vec[i] + vec[i + size];
    }
}

float sum_vec(std::vector<float>& vec) {
    size_t size = vec.size();

    // Ensure size is power of 2 for reduction
    size_t original_size = size;
    size_t padded_size = 1;
    while (padded_size < size) {
        padded_size <<= 1;
    }

    if (padded_size > size) {
        vec.resize(padded_size, 0.0f);
    }

    float* d_vec;
    cudaMalloc(&d_vec, padded_size * sizeof(float));
    cudaMemcpy(d_vec, vec.data(), padded_size * sizeof(float), cudaMemcpyHostToDevice);

    size_t current_size = padded_size / 2;
    int blockSize = 256;

    while (current_size > 0) {
        int numBlocks = (current_size + blockSize - 1) / blockSize;
        _sum_vec<<<numBlocks, blockSize>>>(d_vec, current_size);
        cudaDeviceSynchronize();
        current_size >>= 1;
    }

    float result;
    cudaMemcpy(&result, d_vec, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);

    // Restore original size if we padded it
    if (padded_size > original_size) {
        vec.resize(original_size);
    }

    return result;
}

__global__ void get_max_val(float* vec, float* max_out, size_t size) {
    extern __shared__ float shared_max[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float local_max = -FLT_MAX;

    for (; i < size; i += blockDim.x * gridDim.x) {
        float val = vec[i];
        if (val > local_max) {
            local_max = val;
        }
    }

    shared_max[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_max[tid + s] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_out, shared_max[0]);
    }
}

__global__ void calc_exp(float* dst, float* src, float max_val, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dst[i] = expf(src[i] - max_val);
    }
}

__global__ void calc_divis(float* dst, float sum, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dst[i] = dst[i] / sum;
    }
}

void softmax_gpu(std::vector<float>& vec) {
    if (vec.empty()) return;

    size_t size = vec.size();
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    float *d_vec, *d_exp, *d_max;
    cudaMalloc(&d_vec, size * sizeof(float));
    cudaMalloc(&d_exp, size * sizeof(float));
    cudaMalloc(&d_max, sizeof(float));

    cudaMemcpy(d_vec, vec.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float initial_max = -FLT_MAX;
    cudaMemcpy(d_max, &initial_max, sizeof(float), cudaMemcpyHostToDevice);

    get_max_val<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_vec, d_max, size);
    cudaMemcpy(&initial_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    calc_exp<<<numBlocks, blockSize>>>(d_exp, d_vec, initial_max, size);
    cudaDeviceSynchronize();

    std::vector<float> exp_vec(size);
    cudaMemcpy(exp_vec.data(), d_exp, size * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = sum_vec(exp_vec);

    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));

    calc_divis<<<numBlocks, blockSize>>>(d_exp, sum, size);
    cudaDeviceSynchronize();

    cudaMemcpy(vec.data(), d_exp, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_exp);
    cudaFree(d_max);
    cudaFree(d_sum);
}

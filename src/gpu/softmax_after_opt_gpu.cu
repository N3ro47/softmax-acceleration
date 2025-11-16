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

static __global__ void get_max_val(float* vec, float* max_out, size_t size) {
    extern __shared__ float shared_max[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float local_max = -FLT_MAX;

    for (; i < size; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, vec[i]);
    }

    shared_max[tid] = local_max;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(max_out, shared_max[0]);
    }
}


static __global__ void sum_reduce(float* vec, float* sum_out, size_t size) {
    extern __shared__ float shared_sum[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;

    for (; i < size; i += blockDim.x * gridDim.x) {
        local_sum += vec[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(sum_out, shared_sum[0]);
    }
}

static __global__ void calc_exp(float* dst, float* src, float max_val, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dst[i] = expf(src[i] - max_val);
    }
}

static __global__ void calc_divis(float* dst, float sum, size_t size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        dst[i] = dst[i] / sum;
    }
}

void softmax_gpu_opt(std::vector<float>& vec) {
    if (vec.empty()) return;

    const size_t size = vec.size();


    const int threadsPerBlock = 256;
    const int maxBlocks = 1024; 
    const int numBlocks = min(maxBlocks, (int)((size + threadsPerBlock - 1) / threadsPerBlock));

    float *d_vec = nullptr, *d_exp = nullptr, *d_max = nullptr, *d_sum = nullptr;


    cudaError_t err;
    err = cudaMalloc(&d_vec, size * sizeof(float));
    if (err != cudaSuccess) { /* handle error */ }

    err = cudaMalloc(&d_exp, size * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_vec);
        return;
    }

    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_vec);
        cudaFree(d_exp);
        return;
    }

    err = cudaMalloc(&d_sum, sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_vec);
        cudaFree(d_exp);
        cudaFree(d_max);
        return;
    }

    cudaMemcpy(d_vec, vec.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float initial_max = -FLT_MAX;
    cudaMemcpy(d_max, &initial_max, sizeof(float), cudaMemcpyHostToDevice);

    get_max_val<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_vec, d_max, size);
    cudaDeviceSynchronize();

    float max_val;
    cudaMemcpy(&max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);

    calc_exp<<<numBlocks, threadsPerBlock>>>(d_exp, d_vec, max_val, size);
    cudaDeviceSynchronize();

    float initial_sum = 0.0f;
    cudaMemcpy(d_sum, &initial_sum, sizeof(float), cudaMemcpyHostToDevice);

    sum_reduce<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_exp, d_sum, size);
    cudaDeviceSynchronize();

    float sum_val;
    cudaMemcpy(&sum_val, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    calc_divis<<<numBlocks, threadsPerBlock>>>(d_exp, sum_val, size);
    cudaDeviceSynchronize();

    cudaMemcpy(vec.data(), d_exp, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_vec);
    cudaFree(d_exp);
    cudaFree(d_max);
    cudaFree(d_sum);
}

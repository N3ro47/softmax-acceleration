// src/gpu/softmax_gpu_thrust.cu

#include "softmax.h"
#include <vector>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

struct exp_functor {
    const float max_val;
    exp_functor(float _max_val) : max_val(_max_val) {}
    __host__ __device__ float operator()(const float& x) const { return expf(x - max_val); }
};

struct division_functor {
    const float sum_val;
    division_functor(float _sum_val) : sum_val(_sum_val) {}
    __host__ __device__ float operator()(const float& x) const { return x / sum_val; }
};

void softmax_gpu_thrust(std::vector<float>& vec) {
    if (vec.empty()) return;

    thrust::device_vector<float> d_vec(vec.begin(), vec.end());

    float max_val = *thrust::max_element(d_vec.begin(), d_vec.end());

    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), exp_functor(max_val));

    float sum_val = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

    if (sum_val > 0.0f) {
        thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), division_functor(sum_val));
    }

    thrust::copy(d_vec.begin(), d_vec.end(), vec.begin());
}
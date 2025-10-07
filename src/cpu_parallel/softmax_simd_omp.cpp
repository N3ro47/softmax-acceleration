#include "softmax.h"
#include "simd_helpers.h"
#include <vector>
#include <algorithm>
#include <omp.h>

void softmax_simd_omp(std::vector<float>& vec) {
    if (vec.empty()) return;
    float* data = vec.data();
    const size_t n = vec.size();

    // Step 1: Find the maximum value in parallel
    float max_val = data[0];
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 1; i < n; ++i) {
        max_val = std::fmax(max_val, data[i]);
    }

    // Step 2: Calculate exponents and sum using both OpenMP and SIMD.
    // The loop is parallelized across threads, and each thread uses SIMD instructions.
    float sum_exp = 0.0f;
    const __m256 max_v = _mm256_set1_ps(max_val);

    #pragma omp parallel for reduction(+:sum_exp)
    for (size_t i = 0; i < n / 8; ++i) {
        size_t index = i * 8;
        __m256 vals = load_ps(data + index);
        __m256 shifted = sub_ps(vals, max_v);
        __m256 exps = exp_approx_ps(shifted);
        store_ps(data + index, exps);
        sum_exp += hsum_ps(exps); // Each thread accumulates its own partial sum
    }

    // Handle the remainder elements that don't fit in a SIMD register.
    // This small loop can also be parallelized, though the overhead might be significant
    // if the remainder is small. It is included for completeness.
    #pragma omp parallel for reduction(+:sum_exp)
    for (size_t i = (n / 8) * 8; i < n; ++i) {
        float e = std::exp(data[i] - max_val);
        data[i] = e;
        sum_exp += e;
    }

    // Step 3: Normalize the vector in parallel using SIMD.
    if (sum_exp > 0.0f) {
        const float recip_sum = 1.0f / sum_exp;
        const __m256 recip_v = _mm256_set1_ps(recip_sum);

        #pragma omp parallel for
        for (size_t i = 0; i < n / 8; ++i) {
            size_t index = i * 8;
            __m256 exps = load_ps(data + index);
            store_ps(data + index, mul_ps(exps, recip_v));
        }

        // Handle remainder elements.
        #pragma omp parallel for
        for (size_t i = (n / 8) * 8; i < n; ++i) {
            data[i] *= recip_sum;
        }
    }
}



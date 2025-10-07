#include "softmax.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

void softmax_naive_omp(std::vector<float>& vec) {
    if (vec.empty()) return;

    // Step 1: Find the maximum value in parallel.
    float max_val = vec[0];
    #pragma omp parallel for reduction(max:max_val)
    for (size_t i = 1; i < vec.size(); ++i) {
        max_val = std::fmax(max_val, vec[i]);
    }

    // Step 2: Calculate exponents and their sum in parallel.
    float sum_exp = 0.0f;
    #pragma omp parallel for reduction(+:sum_exp)
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = std::exp(vec[i] - max_val);
        sum_exp += vec[i];
    }

    // Step 3: Normalize the vector in parallel.
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); ++i) {
        if (sum_exp > 0.0f) {
            vec[i] /= sum_exp;
        }
    }
}



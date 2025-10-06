#include "softmax.h"
#include <cmath>
#include <numeric>
#include <algorithm>

void softmax_foolish_handcoding_cpu(std::vector<float>& vec) {
    if (vec.empty()) return;

    float* data = vec.data();
    size_t n = vec.size();

    float max_val = data[0];
    for (size_t i = 1; i < n; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float exp_val = std::exp(data[i] - max_val);
        data[i] = exp_val;
        sum_exp += exp_val;
    }

    if (sum_exp > 0.0f) {
        float recip = 1.0f / sum_exp;
        for (size_t i = 0; i < n; ++i) {
            data[i] *= recip;
        }
    }
}
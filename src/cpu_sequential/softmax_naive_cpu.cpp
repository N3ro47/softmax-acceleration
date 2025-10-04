#include "softmax.h"
#include <cmath>
#include <numeric>
#include <algorithm>

void softmax_naive_cpu(std::vector<float>& vec) {
    if (vec.empty()) return;

    float max_val = *std::max_element(vec.begin(), vec.end());
    float sum_exp = 0.0f;

    for (float& val : vec) {
        val = std::exp(val - max_val);
        sum_exp += val;
    }

    for (float& val : vec) {
        val /= sum_exp;
    }
}

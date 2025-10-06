#include "softmax.h"
#include "simd_helpers.h"

void softmax_simd_cpu(std::vector<float>& vec) {
    if (vec.empty()) return;
    float* data = vec.data();
    size_t n = vec.size();

    float max_val = data[0];
    for (size_t i = 1; i < n; ++i) max_val = std::fmax(max_val, data[i]);

    float sum_exp = 0.0f;
    size_t i = 0;
    __m256 max_v = _mm256_set1_ps(max_val);
    for (; i + 8 <= n; i += 8) {
        __m256 vals = load_ps(data + i);
        __m256 shifted = sub_ps(vals, max_v);
        __m256 exps = exp_approx_ps(shifted);
        store_ps(data + i, exps);
        sum_exp += hsum_ps(exps);
    }
    for (; i < n; ++i) {
        float e = std::exp(data[i] - max_val);
        data[i] = e;
        sum_exp += e;
    }

    if (sum_exp > 0.0f) {
        __m256 recip_v = _mm256_set1_ps(1.0f / sum_exp);
        for (i = 0; i + 8 <= n; i += 8) {
            __m256 exps = load_ps(data + i);
            store_ps(data + i, mul_ps(exps, recip_v));
        }
        for (; i < n; ++i) data[i] *= 1.0f / sum_exp;
    }
}
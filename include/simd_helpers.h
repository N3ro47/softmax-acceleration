#ifndef SIMD_HELPERS_H
#define SIMD_HELPERS_H

#include <immintrin.h>
#include <cmath>

inline __m256 load_ps(float* ptr) { return _mm256_loadu_ps(ptr); }
inline void store_ps(float* ptr, __m256 vec) { _mm256_storeu_ps(ptr, vec); }
inline __m256 add_ps(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
inline __m256 sub_ps(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
inline __m256 mul_ps(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
inline __m256 fmadd_ps(__m256 a, __m256 b, __m256 c) { return _mm256_fmadd_ps(a, b, c); }

inline float hsum_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    
    return _mm_cvtss_f32(sum);
}

inline __m256 exp_approx_ps(__m256 x) {
    __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    __m256 inv_log2ef = _mm256_set1_ps(0.693147180559945f);
    __m256 p0 = _mm256_set1_ps(1.9875691500E-4f);
    __m256 p1 = _mm256_set1_ps(1.3981999507E-3f);
    __m256 p2 = _mm256_set1_ps(8.3334519073E-3f);
    __m256 p3 = _mm256_set1_ps(4.1665795894E-2f);
    __m256 p4 = _mm256_set1_ps(1.6666665459E-1f);
    __m256 p5 = _mm256_set1_ps(5.0000001201E-1f);
    __m256 one = _mm256_set1_ps(1.0f);

    __m256 fx = mul_ps(x, log2ef);
    fx = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 z = mul_ps(fx, inv_log2ef);
    x = sub_ps(x, z);
    z = mul_ps(x, x);

    __m256 y = p0;
    y = fmadd_ps(y, x, p1);
    y = fmadd_ps(y, x, p2);
    y = fmadd_ps(y, x, p3);
    y = fmadd_ps(y, x, p4);
    y = fmadd_ps(y, x, p5);
    y = mul_ps(y, z);
    y = add_ps(y, x);
    y = add_ps(y, one);

    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(127));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = mul_ps(y, pow2n);
    return y;
}

#endif  // SIMD_HELPERS_H
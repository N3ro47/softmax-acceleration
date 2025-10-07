#include "gtest/gtest.h"
#include "softmax.h"
#include "utils.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace {

void expect_vectors_close(const std::vector<float>& a, const std::vector<float>& b, float atol = 1e-5f, float rtol = 1e-5f) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float tol = atol + rtol * std::max(std::fabs(a[i]), std::fabs(b[i]));
        ASSERT_LE(diff, tol) << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i];
    }
}

void expect_softmax_properties(const std::vector<float>& v) {
    if (v.empty()) return;
    float sum = 0.0f;
    for (float x : v) {
        ASSERT_GE(x, 0.0f);
        sum += x;
    }
    ASSERT_NEAR(sum, 1.0f, 1e-4f);
}

std::vector<float> random_vector(size_t n, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

} // namespace

TEST(SoftmaxNaiveCpu, HandlesEmptyVector) {
    std::vector<float> v;
    softmax_naive_cpu(v);
    EXPECT_TRUE(v.empty());
}

TEST(SoftmaxReference, DeterministicSmallVector) {
    std::vector<float> v = {0.0f, 1.0f, 2.0f, -3.0f};
    std::vector<float> ref = v;
    softmax_naive_cpu(ref);
    expect_softmax_properties(ref);
}

TEST(SoftmaxImplementations, RandomVectorsAgreeWithNaiveCpu) {
    struct Impl { const char* name; void (*fn)(std::vector<float>&); float atol; float rtol; };
    std::vector<Impl> impls = {
        {"softmax_simd_cpu", softmax_simd_cpu, 1e-5f, 1e-5f},
        {"softmax_foolish_handcoding_cpu", softmax_foolish_handcoding_cpu, 1e-5f, 1e-5f},
        {"softmax_fused_simd_cpu", softmax_fused_simd_cpu, 2e-5f, 2e-5f},
    };
#ifdef HAVE_SOFTMAX_OMP
    impls.push_back({"softmax_naive_omp", softmax_naive_omp, 2e-5f, 2e-5f});
    impls.push_back({"softmax_simd_omp", softmax_simd_omp, 2e-5f, 2e-5f});
#endif
#ifdef HAVE_ONEDNN
    impls.push_back({"softmax_onednn_cpu", softmax_onednn_cpu, 1e-5f, 1e-5f});
#endif

    for (size_t n : {0ULL, 1ULL, 3ULL, 16ULL, 1024ULL, 65536ULL}) {
        std::vector<float> base = random_vector(n);
        std::vector<float> ref = base;
        softmax_naive_cpu(ref);
        expect_softmax_properties(ref);

        for (const auto& impl : impls) {
            SCOPED_TRACE(::testing::Message() << "impl=" << impl.name << ", n=" << n);
            std::vector<float> w = base;
            impl.fn(w);
            expect_softmax_properties(w);
            expect_vectors_close(w, ref, impl.atol, impl.rtol);
        }
    }
}


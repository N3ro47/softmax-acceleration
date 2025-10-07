#include "benchmark/benchmark.h"
#include "softmax.h"
#include "utils.h"
#include <vector>
#include <string>
#include <cstdlib>
#ifdef HAVE_SOFTMAX_OMP
#include <omp.h>
#endif

class SoftmaxBench : public benchmark::Fixture {
public:
    void SetUp(::benchmark::State& state) {
        #ifdef HAVE_SOFTMAX_OMP
        if (const char* t = std::getenv("SOFTMAX_OMP_THREADS")) {
            int n = std::atoi(t);
            if (n > 0) omp_set_num_threads(n);
        } else if (const char* t2 = std::getenv("OMP_NUM_THREADS")) {
            int n2 = std::atoi(t2);
            if (n2 > 0) omp_set_num_threads(n2);
        }
        #endif
        int64_t vector_size = state.range(0);
        std::string filename = "data/vector_" + std::to_string(vector_size) + ".bin";
        if (!read_vector_from_file(filename, data)) {
            state.SkipWithError("Failed to read test data file.");
        }
    }

    void TearDown(::benchmark::State& state) {
        data.clear();
    }

protected:
    std::vector<float> data;
};


BENCHMARK_DEFINE_F(SoftmaxBench, NaiveCPU)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_naive_cpu(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, FoolishHandCoding)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_foolish_handcoding_cpu(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, SimdCpu)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_simd_cpu(test_data);
    }
}

#ifdef HAVE_SOFTMAX_OMP
BENCHMARK_DEFINE_F(SoftmaxBench, NaiveOMP)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_naive_omp(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, SimdOMP)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_simd_omp(test_data);
    }
}
#endif

BENCHMARK_DEFINE_F(SoftmaxBench, FusedSimdCpu)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_fused_simd_cpu(test_data);
    }
}

#ifdef HAVE_ONEDNN
BENCHMARK_DEFINE_F(SoftmaxBench, OneDNN_CPU)(benchmark::State& state) {
    for (auto _ : state) {
        state.PauseTiming();
        std::vector<float> test_data = data;
        state.ResumeTiming();
        softmax_onednn_cpu(test_data);
    }
}
#endif

#define REGISTER_SOFTMAX_BENCHMARK(name) \
    BENCHMARK_REGISTER_F(SoftmaxBench, name) \
        ->Arg(1024) \
        ->Arg(4096) \
        ->Arg(16384) \
        ->Arg(65536) \
        ->Arg(262144) \
        ->Arg(524288) \
        ->Arg(1048576) \
        ->Arg(2097152) \
        ->Arg(4194304) \
        ->Arg(8388608) \
        ->Arg(16777216) \
        ->Arg(33554432) \
        ->Arg(67108864) \
        ->Arg(134217728) \
        ->Arg(268435456) \
        ->Unit(benchmark::kMillisecond)

REGISTER_SOFTMAX_BENCHMARK(NaiveCPU);
REGISTER_SOFTMAX_BENCHMARK(SimdCpu);
 #ifdef HAVE_SOFTMAX_OMP
REGISTER_SOFTMAX_BENCHMARK(NaiveOMP);
REGISTER_SOFTMAX_BENCHMARK(SimdOMP);
 #endif
REGISTER_SOFTMAX_BENCHMARK(FoolishHandCoding);
REGISTER_SOFTMAX_BENCHMARK(FusedSimdCpu);
#ifdef HAVE_ONEDNN
REGISTER_SOFTMAX_BENCHMARK(OneDNN_CPU);
#endif


BENCHMARK_MAIN();
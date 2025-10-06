#include "benchmark/benchmark.h"
#include "softmax.h"
#include "utils.h"
#include <vector>
#include <string>

class SoftmaxBench : public benchmark::Fixture {
public:
    void SetUp(::benchmark::State& state) {
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
        std::vector<float> test_data = data;
        softmax_naive_cpu(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, FoolishHandCoding)(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<float> test_data = data;
        softmax_foolish_handcoding_cpu(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, SimdCpu)(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<float> test_data = data;
        softmax_simd_cpu(test_data);
    }
}

BENCHMARK_DEFINE_F(SoftmaxBench, FusedSimdCpu)(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<float> test_data = data;
        softmax_fused_simd_cpu(test_data);
    }
}


BENCHMARK_REGISTER_F(SoftmaxBench, NaiveCPU)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576)
    ->Arg(2097152)
    ->Arg(4194304)
    ->Arg(8388608)
    ->Arg(16777216)
    ->Arg(33554432)
    ->Arg(67108864)
    ->Arg(134217728)
    ->Arg(268435456)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SoftmaxBench, FoolishHandCoding)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576)
    ->Arg(2097152)
    ->Arg(4194304)
    ->Arg(8388608)
    ->Arg(16777216)
    ->Arg(33554432)
    ->Arg(67108864)
    ->Arg(134217728)
    ->Arg(268435456)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SoftmaxBench, SimdCpu)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576)
    ->Arg(2097152)
    ->Arg(4194304)
    ->Arg(8388608)
    ->Arg(16777216)
    ->Arg(33554432)
    ->Arg(67108864)
    ->Arg(134217728)
    ->Arg(268435456)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(SoftmaxBench, FusedSimdCpu)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Arg(524288)
    ->Arg(1048576)
    ->Arg(2097152)
    ->Arg(4194304)
    ->Arg(8388608)
    ->Arg(16777216)
    ->Arg(33554432)
    ->Arg(67108864)
    ->Arg(134217728)
    ->Arg(268435456)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
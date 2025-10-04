#include "benchmark/benchmark.h"
#include "softmax.h"
#include "utils.h"
#include <vector>
#include <string>

// This fixture reads the data file once for all benchmarks using it.
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


// Benchmark for the naive CPU implementation
BENCHMARK_DEFINE_F(SoftmaxBench, NaiveCPU)(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<float> test_data = data;
        softmax_naive_cpu(test_data);
    }
}

// Register the benchmark to run for a range of vector sizes
BENCHMARK_REGISTER_F(SoftmaxBench, NaiveCPU)
    ->Arg(1024)
    ->Arg(4096)
    ->Arg(16384)
    ->Arg(65536)
    ->Arg(262144)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
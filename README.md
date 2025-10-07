# Project: Softmax Acceleration

This project analyzes and benchmarks multiple implementations of the softmax algorithm.

## Prerequisites

- A recent C++17 compiler (GCC/Clang)
- CMake >= 3.15 and Ninja
- Python 3 (for data generation script)
- Optional: OpenMP for parallel CPU variants
- Optional: oneDNN (DNNL) for the oneDNN implementation

## Project layout

- `include/` — public headers like `softmax.h`
- `src/cpu_sequential/` — sequential CPU implementations
- `src/cpu_parallel/` — OpenMP-based CPU implementations (built if OpenMP is found)
- `benchmarks/` — Google Benchmark driver and registrations
- `tests/` — GoogleTest unit tests
- `common/` — shared utilities
- `scripts/` — helpers (e.g., `generate_data.py`)
- `data/` — generated input vectors for benchmarks

## Quick Start

1.  **Configure & Build:**
    This will download dependencies and compile the code using CMake/Ninja.
    ```bash
    make build
    ```

2.  **Run Benchmarks:**
    This command generates test data for various sizes and runs all C++ benchmarks.
    ```bash
    make benchmark
    ```

3.  **Clean Up:**
    ```bash
    make clean
    ```

## Available Make targets

- `make configure` — run CMake configuration (called automatically by `make build`)
- `make build` — build all targets
- `make benchmark` or `make bench` — generate data and run benchmarks
- `make test` — build and run unit tests via `ctest`
- `make clean` — remove the build directory and generated data files

## Tests

Run the unit tests (GoogleTest):
```bash
make test
```
This builds the test binary and runs it via `ctest`.

## Benchmarks

Run benchmarks (both commands are equivalent):
```bash
make benchmark
# or
make bench
```
- Data files are generated automatically under `data/` for the preset sizes.
- To control OpenMP threads during benchmarking, you can set `SOFTMAX_OMP_THREADS` or `OMP_NUM_THREADS`.
  ```bash
  SOFTMAX_OMP_THREADS=8 make bench
  ```
- The benchmark sizes are defined in `Makefile` and registered in `benchmarks/main_bench.cpp`.

## Adding a new softmax implementation

Follow these steps to add a new implementation and integrate it into builds, benchmarks, and tests.

1) Declare the function
- Add the declaration to `include/softmax.h` (choose the appropriate section, e.g., CPU, OpenMP, or oneDNN/GPU in the future):
  ```cpp
  void softmax_my_awesome_impl(std::vector<float>& vec);
  ```

2) Implement the function
- Create the implementation file in the appropriate directory:
  - Sequential CPU: `src/cpu_sequential/`
  - OpenMP parallel CPU: `src/cpu_parallel/`
- Example path:
  - `src/cpu_sequential/softmax_my_awesome_impl.cpp`

3) Register the file in the build
- Edit `CMakeLists.txt`:
  - If it is a sequential CPU implementation, add the `.cpp` to the `SOFTMAX_SOURCES` list.
  - If it is an OpenMP variant, add the `.cpp` to the `softmax_omp` target sources (inside the `if (OpenMP_CXX_FOUND)` block).

4) Add a benchmark entry
- Open `benchmarks/main_bench.cpp` and:
  - Define a benchmark body similar to existing ones using `BENCHMARK_DEFINE_F(SoftmaxBench, Name)` and call your function inside it.
  - Register it with the helper macro so it runs at the preset sizes:
    ```cpp
    REGISTER_SOFTMAX_BENCHMARK(My_Awesome_Impl);
    ```
  - Tip: Pick a concise, descriptive benchmark name to replace `My_Awesome_Impl`.

5) Add the implementation to tests (recommended)
- Open `tests/softmax_tests.cpp` and add your implementation to the `impls` list so it is compared against the reference naive CPU version:
  ```cpp
  {"softmax_my_awesome_impl", softmax_my_awesome_impl, 1e-5f, 1e-5f},
  ```
- Adjust tolerances if needed for numerical differences.

6) Build and run
```bash
make build
make test
make bench
```

Notes
- If oneDNN is installed and detected, an additional oneDNN-based implementation and benchmarks will be built automatically.
- If OpenMP is available, OpenMP-based variants will also be enabled.

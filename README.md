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

## JAX vs hand-written CUDA — the `@jax.jit` experiment

The CUDA kernels in `src/gpu/` were written by my friend (Michał) as hand-written, multi-kernel implementations. The "optimized version" (`softmax_gpu_opt`) launches 4 separate kernels with `cudaDeviceSynchronize()` between each step (find max → exp → sum → divide).

Before I even started, my hypothesis was that a 4-line Python function with `@jax.jit`:

```python
@jax.jit
def softmax_jax_manual(x):
    x_max = jnp.max(x)
    e_x = jnp.exp(x - x_max)
    return e_x / jnp.sum(e_x)
```

…would blow away Michał's "optimized" CUDA implementation, because XLA fuses the entire computation into a single GPU kernel zero intermediate global memory writes, zero sync barriers. Of course there is limit from what can you expect your uni friend to do writting cuda for the first time at uni.

**It did. By up to 16×.**

### C++ CUDA results (`make bench`, RTX 3060 Ti)

```
SoftmaxBench/Gpu_opt/1024          0.139 ms
SoftmaxBench/Gpu_opt/65536         0.199 ms
SoftmaxBench/Gpu_opt/1048576       1.02  ms
SoftmaxBench/Gpu_opt/8388608       6.67  ms
SoftmaxBench/Gpu_opt/67108864     43.3   ms
SoftmaxBench/Gpu_opt/268435456   173     ms
```

### JAX results (`make benchmark-jax`)

```
          Size     jax.nn.softmax (ms)     manual softmax (ms)
--------------------------------------------------------------
         1,024      0.2134 ± 0.0075        0.0518 ± 0.0021
        65,536      0.2298 ± 0.0187        0.0499 ± 0.0022
     1,048,576      0.2696 ± 0.0192        0.1094 ± 0.0026
     8,388,608      0.7656 ± 0.0123        0.3984 ± 0.0097
    67,108,864      5.3576 ± 0.0870        2.6608 ± 0.0274
   268,435,456     20.9255 ± 0.0258       10.4152 ± 0.0392
```

### Head-to-head: JAX manual vs C++ GPU_opt

| Size | C++ GPU_opt | JAX `@jax.jit` | Speedup |
|---:|---:|---:|---:|
| 1K | 0.139 ms | 0.052 ms | **2.6×** |
| 64K | 0.199 ms | 0.050 ms | **3.0×** |
| 1M | 1.02 ms | 0.100 ms | **10.2×** |
| 8M | 6.67 ms | 0.399 ms | **16.7×** |
| 67M | 43.3 ms | 2.74 ms | **15.8×** |
| 268M | 173 ms | 10.8 ms | **16.0×** |

XLA compiles the whole softmax into a single fused Triton kernel, while the C++ version pays for 4 kernel launches + syncs + intermediate global memory traffic. To close the gap in C++ you'd need to implement online softmax (single-pass fused max+sum+normalize).



```bash
make setup-jax       # one-time: creates .venv with JAX + CUDA 12
make benchmark-jax   # runs JAX GPU benchmarks
```

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

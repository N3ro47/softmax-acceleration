import jax
import jax.numpy as jnp
import numpy as np
import argparse
import time

# JIT-compile the softmax function for GPU execution
@jax.jit
def softmax_jax(x):
    e_x = jnp.exp(x - jnp.max(x))
    return e_x / e_x.sum()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run JAX softmax implementation.")
    parser.add_argument('--input', type=str, required=True, help='Path to binary data file.')
    args = parser.parse_args()

    # Load data from the same file as C++
    data = np.fromfile(args.input, dtype=np.float32)
    print(f"Loaded vector of size {data.shape[0]} from {args.input}")
    
    # JAX works on device arrays
    x_device = jax.device_put(data)

    # Warm-up run to compile
    print("Performing warm-up run...")
    result_warmup = softmax_jax(x_device).block_until_ready()

    # Timed run
    print("Performing timed run...")
    start_time = time.perf_counter()
    result = softmax_jax(x_device).block_until_ready()
    end_time = time.perf_counter()

    print(f"JAX softmax execution time: {(end_time - start_time) * 1000:.4f} ms")
    print("First 5 results:", result[:5])

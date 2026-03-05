#!/usr/bin/env python3

import argparse
import time
import sys

import jax
import jax.numpy as jnp
import numpy as np


def softmax_jax_builtin(x: jax.Array) -> jax.Array:
    return jax.nn.softmax(x, axis=-1)


@jax.jit
def softmax_jax_manual(x: jax.Array) -> jax.Array:
    x_max = jnp.max(x)
    e_x = jnp.exp(x - x_max)
    return e_x / jnp.sum(e_x)

DEFAULT_SIZES = [
    1024, 4096, 16384, 65536, 262144, 524288,
    1048576, 2097152, 4194304, 8388608,
    16777216, 33554432, 67108864, 134217728, 268435456,
]


def bench_fn(fn, x, warmup: int, repeats: int) -> tuple[float, float]:
    for _ in range(warmup):
        out = fn(x)
        out.block_until_ready()

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(x)
        out.block_until_ready()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    arr = np.array(times)
    return float(np.mean(arr)), float(np.std(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="JAX CUDA softmax benchmarks")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
        help="Vector sizes to benchmark (default: same as C++ benchmarks)",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--repeats", type=int, default=20, help="Timed iterations")
    args = parser.parse_args()

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print("ERROR: No CUDA GPU found by JAX. Exiting.", file=sys.stderr)
        sys.exit(1)

    gpu = gpu_devices[0]
    print(f"JAX version : {jax.__version__}")
    print(f"Device      : {gpu}")
    print(f"Warm-up     : {args.warmup} iters")
    print(f"Repeats     : {args.repeats} iters")
    print()

    header = f"{'Size':>14s}  {'jax.nn.softmax (ms)':>22s}  {'manual softmax (ms)':>22s}"
    print(header)
    print("-" * len(header))

    key = jax.random.PRNGKey(42)

    for size in args.sizes:
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=(size,), minval=-10.0, maxval=10.0)
        x.block_until_ready()

        mean_b, std_b = bench_fn(softmax_jax_builtin, x, args.warmup, args.repeats)

        mean_m, std_m = bench_fn(softmax_jax_manual, x, args.warmup, args.repeats)

        print(
            f"{size:>14,d}  "
            f"{mean_b:>10.4f} ± {std_b:<8.4f}  "
            f"{mean_m:>10.4f} ± {std_m:<8.4f}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()

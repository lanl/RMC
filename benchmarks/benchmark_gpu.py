#!/usr/bin/env python3
"""GPU (Metal) performance benchmark for RMC on Apple M4 Pro."""

import time
import os

# Force Metal (GPU)
os.environ['JAX_PLATFORMS'] = 'metal'

import jax
import jax.numpy as jnp
import numpy as np

print("="*70)
print("RMC GPU (Metal) Performance Benchmark - Apple M4 Pro")
print("="*70)

# Import after setting Metal
from rmc.utils.kernels import cdist, cdistSQ, RBF_Gramm
from rmc.modules.sampler import accept_Metropolis

print(f"\nJAX Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print()

def benchmark(func, *args, n_runs=10, warmup=5, **kwargs):
    """Simple benchmark function with GPU sync."""
    # Warmup (more runs for GPU)
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    # Time
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        elif hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)

# Test configurations
configs = [
    ("Small", 100, 10),
    ("Medium", 1000, 50),
    ("Large", 5000, 100),
]

print("Running benchmarks on Apple M4 Pro GPU...\n")

for name, n_samples, dim in configs:
    print(f"{'-'*70}")
    print(f"Configuration: {name} (n={n_samples}, d={dim})")
    print(f"{'-'*70}")

    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (n_samples, dim))
    y = jax.random.normal(key, (n_samples, dim))

    # 1. cdist
    mean_t, std_t = benchmark(cdist, x, y)
    print(f"  cdist:        {mean_t:8.3f} ± {std_t:6.3f} ms")

    # 2. cdistSQ
    mean_t, std_t = benchmark(cdistSQ, x, y)
    print(f"  cdistSQ:      {mean_t:8.3f} ± {std_t:6.3f} ms")

    # 3. RBF_Gramm
    mean_t, std_t = benchmark(RBF_Gramm, x)
    print(f"  RBF_Gramm:    {mean_t:8.3f} ± {std_t:6.3f} ms")

    # 4. Metropolis acceptance
    E_old = jax.random.normal(key, (n_samples,))
    E_new = jax.random.normal(key, (n_samples,))
    mean_t, std_t = benchmark(accept_Metropolis, key, E_old, E_new)
    print(f"  Metropolis:   {mean_t:8.3f} ± {std_t:6.3f} ms")

    # 5. Matrix operations
    @jax.jit
    def matmul_op(x, y):
        return jnp.dot(x, y.T)

    mean_t, std_t = benchmark(matmul_op, x, y)
    print(f"  Matrix mult:  {mean_t:8.3f} ± {std_t:6.3f} ms")
    print()

print(f"{'='*70}")
print("GPU Benchmark Complete!")
print("="*70)

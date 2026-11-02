#!/usr/bin/env python3
"""Quick performance benchmark for RMC on Apple M4 Pro."""

import time
import os

# Force CPU first
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np

print("="*70)
print("RMC Quick Performance Benchmark - Apple M4 Pro")
print("="*70)

# Import after setting CPU
from rmc.utils.kernels import cdist, cdistSQ, RBF_Gramm, RBF
from rmc.modules.sampler import accept_Metropolis

print(f"\nJAX Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print()

def benchmark(func, *args, n_runs=10, warmup=3, **kwargs):
    """Simple benchmark function."""
    # Warmup
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

results = {}

for name, n_samples, dim in configs:
    print(f"\n{'-'*70}")
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

    results[name] = {
        'n_samples': n_samples,
        'dim': dim,
        'cdist': mean_t
    }

print(f"\n{'='*70}")
print("CPU Benchmark Complete!")
print("="*70)
print()

# Now try GPU if available
try:
    # Clear environment
    if 'JAX_PLATFORMS' in os.environ:
        del os.environ['JAX_PLATFORMS']

    # Force reload JAX with GPU
    import importlib
    import sys

    # Remove jax modules
    mods_to_remove = [k for k in sys.modules.keys() if k.startswith('jax')]
    for mod in mods_to_remove:
        del sys.modules[mod]

    print("\nAttempting to use GPU (Metal)...")
    print("Note: This requires restarting Python. Run with --gpu flag separately.")
    print()

except Exception as e:
    print(f"GPU setup error: {e}")
    print()

print("To run GPU benchmarks, restart Python and set JAX_PLATFORMS=metal")
print()

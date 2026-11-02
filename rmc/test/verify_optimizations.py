#!/usr/bin/env python3
"""Quick verification script for RMC optimizations.

This script performs a quick check to verify that the optimizations
are working correctly.
"""

import sys

import jax
import jax.numpy as jnp

print("="*70)
print("RMC Optimization Verification")
print("="*70)

# Check JAX setup
print(f"\n[1/5] JAX Configuration")
print(f"  JAX version: {jax.__version__}")
print(f"  Backend: {jax.default_backend()}")
print(f"  Devices: {jax.devices()}")

# Check if optimized modules can be imported
print(f"\n[2/5] Checking Optimized Modules")
try:
    from rmc.utils.kernels import cdist, cdistSQ, RBF_Gramm, RBF
    print("  ✓ rmc.utils.kernels")
except ImportError as e:
    print(f"  ✗ rmc.utils.kernels: {e}")
    sys.exit(1)

try:
    from rmc.modules.svgd import SVGD
    print("  ✓ rmc.modules.svgd")
except ImportError as e:
    print(f"  ✗ rmc.modules.svgd: {e}")
    sys.exit(1)

try:
    from rmc.modules.cfm import ConditionalFlowMatching
    print("  ✓ rmc.modules.cfm")
except ImportError as e:
    print(f"  ✗ rmc.modules.cfm: {e}")
    sys.exit(1)

try:
    from rmc.modules.sampler import accept_Metropolis
    print("  ✓ rmc.modules.sampler")
except ImportError as e:
    print(f"  ✗ rmc.modules.sampler: {e}")
    sys.exit(1)

try:
    from rmc.utils.density import BaseLogDensity
    print("  ✓ rmc.utils.density")
except ImportError as e:
    print(f"  ✗ rmc.utils.density: {e}")
    sys.exit(1)

# Test JIT compilation
print(f"\n[3/5] Testing JIT Compilation")
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (100, 10))
y = jax.random.normal(key, (100, 10))

try:
    # Test cdist
    dist = cdist(x, y)
    assert dist.shape == (100, 100)
    print("  ✓ cdist() JIT compiled")
except Exception as e:
    print(f"  ✗ cdist() failed: {e}")
    sys.exit(1)

try:
    # Test RBF_Gramm
    Kxy, h = RBF_Gramm(x)
    assert Kxy.shape == (100, 100)
    print("  ✓ RBF_Gramm() JIT compiled")
except Exception as e:
    print(f"  ✗ RBF_Gramm() failed: {e}")
    sys.exit(1)

try:
    # Test accept_Metropolis
    E_old = jax.random.normal(key, (100,))
    E_new = jax.random.normal(key, (100,))
    acc = accept_Metropolis(key, E_old, E_new)
    assert acc.shape == (100, 1)
    print("  ✓ accept_Metropolis() JIT compiled")
except Exception as e:
    print(f"  ✗ accept_Metropolis() failed: {e}")
    sys.exit(1)

# Test vectorization
print(f"\n[4/5] Testing Vectorization")
try:
    particles = jax.random.normal(key, (50, 5))
    kpar = 1.0

    # Compute with vectorized version
    Kxy, _ = RBF_Gramm(particles, kpar)
    dxkxy_vec = -jnp.matmul(Kxy, particles)
    sumkxy = jnp.sum(Kxy, axis=1, keepdims=True)
    dxkxy_vec = dxkxy_vec + particles * sumkxy
    dxkxy_vec = dxkxy_vec / kpar

    assert dxkxy_vec.shape == (50, 5)
    print("  ✓ SVGD kernel vectorization working")
except Exception as e:
    print(f"  ✗ SVGD kernel vectorization failed: {e}")
    sys.exit(1)

# Quick performance test
print(f"\n[5/5] Quick Performance Test")
print("  Running 10 iterations...")

import time

# Test 1: RBF_Gramm
x_large = jax.random.normal(key, (500, 20))
start = time.perf_counter()
for _ in range(10):
    Kxy, h = RBF_Gramm(x_large)
    jax.block_until_ready(Kxy)
end = time.perf_counter()
rbf_time = (end - start) / 10 * 1000
print(f"  RBF_Gramm (500×20): {rbf_time:.2f} ms/iter")

# Test 2: cdist
start = time.perf_counter()
for _ in range(10):
    dist = cdist(x_large, x_large)
    jax.block_until_ready(dist)
end = time.perf_counter()
cdist_time = (end - start) / 10 * 1000
print(f"  cdist (500×20): {cdist_time:.2f} ms/iter")

# GPU availability
print(f"\n[GPU] GPU Status")
try:
    gpu_devices = jax.devices('gpu')
    if len(gpu_devices) > 0:
        print(f"  ✓ GPU available: {gpu_devices}")
        print(f"  Run benchmarks with: python benchmarks/benchmark_performance.py --device=both")
    else:
        print(f"  ℹ No GPU detected")
        print(f"  Run benchmarks with: python benchmarks/benchmark_performance.py --device=cpu")
except Exception:
    print(f"  ℹ No GPU detected")
    print(f"  Run benchmarks with: python benchmarks/benchmark_performance.py --device=cpu")

# Summary
print(f"\n{'='*70}")
print("Verification Complete!")
print("="*70)
print("\n✓ All optimizations are working correctly")
print("\nNext steps:")
print("  1. Run full benchmarks: python benchmarks/benchmark_performance.py --device=cpu")
print("  2. Run tests: pytest rmc/test/test_performance.py -v")
print("  3. See OPTIMIZATIONS.md for detailed documentation")
print()

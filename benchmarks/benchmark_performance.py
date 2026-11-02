#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Performance benchmarks for RMC library.

This module provides benchmarks to test CPU and GPU performance for key RMC
operations including:
- Kernel computations (RBF, distance matrices)
- SVGD operations
- CFM evaluations
- Sampler steps
- Density evaluations

Usage:
    python benchmark_performance.py --device=cpu
    python benchmark_performance.py --device=gpu
    python benchmark_performance.py --device=both
"""

import argparse
import time
from functools import partial
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Configure JAX
jax.config.update("jax_enable_x64", True)


def timeit_jax(func: Callable, *args, n_runs: int = 10, warmup: int = 3, **kwargs) -> Tuple[float, float]:
    """Time a JAX function with proper compilation and blocking.

    Args:
        func: JAX function to benchmark
        *args: Positional arguments to pass to func
        n_runs: Number of timing runs
        warmup: Number of warmup runs
        **kwargs: Keyword arguments to pass to func

    Returns:
        Tuple of (mean_time, std_time) in milliseconds
    """
    # Warmup runs to compile
    for _ in range(warmup):
        result = func(*args, **kwargs)
        jax.block_until_ready(result)

    # Timing runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        jax.block_until_ready(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


class KernelBenchmarks:
    """Benchmarks for kernel operations."""

    @staticmethod
    def bench_cdist(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark Euclidean distance computation."""
        from rmc.utils.kernels import cdist, cdistSQ

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (n_samples, dim))
        y = jax.random.normal(key, (n_samples, dim))

        # Benchmark cdist
        mean_cdist, std_cdist = timeit_jax(cdist, x, y)

        # Benchmark cdistSQ
        mean_cdistSQ, std_cdistSQ = timeit_jax(cdistSQ, x, y)

        return {
            "cdist_mean_ms": mean_cdist,
            "cdist_std_ms": std_cdist,
            "cdistSQ_mean_ms": mean_cdistSQ,
            "cdistSQ_std_ms": std_cdistSQ,
        }

    @staticmethod
    def bench_rbf_gramm(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark RBF Gramm matrix computation."""
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (n_samples, dim))

        mean_time, std_time = timeit_jax(RBF_Gramm, x)

        return {
            "rbf_gramm_mean_ms": mean_time,
            "rbf_gramm_std_ms": std_time,
        }


class SVGDBenchmarks:
    """Benchmarks for SVGD operations."""

    @staticmethod
    def bench_svgd_kernel(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark SVGD kernel computation with vectorized derivative."""
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (n_samples, dim))

        @jax.jit
        def svgd_kernel_optimized(particles, kpar=1.0):
            """Optimized SVGD kernel with vectorization."""
            Kxy, _ = RBF_Gramm(particles, kpar)

            # Vectorized computation
            dxkxy = -jnp.matmul(Kxy, particles)
            sumkxy = jnp.sum(Kxy, axis=1, keepdims=True)
            dxkxy = dxkxy + particles * sumkxy
            dxkxy = dxkxy / kpar

            return Kxy, dxkxy

        mean_time, std_time = timeit_jax(svgd_kernel_optimized, particles)

        return {
            "svgd_kernel_mean_ms": mean_time,
            "svgd_kernel_std_ms": std_time,
        }


class CFMBenchmarks:
    """Benchmarks for Conditional Flow Matching operations."""

    @staticmethod
    def bench_cfm_ucond(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark CFM conditional velocity field evaluations."""
        key = jax.random.PRNGKey(42)
        x1 = jax.random.normal(key, (n_samples, dim))
        x = jax.random.normal(key, (n_samples, dim))
        t = jax.random.uniform(key, (n_samples, 1))

        sigma_min = 0.001

        @jax.jit
        def eval_ucond_ot(x1, x, t):
            """Optimal transport conditional vector field."""
            onemsig = 1.0 - sigma_min
            return (x1 - onemsig * x) / (1.0 - onemsig * t)

        mean_time, std_time = timeit_jax(eval_ucond_ot, x1, x, t)

        return {
            "cfm_ucond_mean_ms": mean_time,
            "cfm_ucond_std_ms": std_time,
        }

    @staticmethod
    def bench_cfm_loss(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark CFM loss computation."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (n_samples, dim))
        sigma_min = 0.001

        @jax.jit
        def compute_cfm_loss(x, key):
            """Simplified CFM loss computation."""
            key, subkey1, subkey2 = jax.random.split(key, 3)
            x0 = jax.random.normal(subkey1, (n_samples, dim))
            t = jax.random.uniform(subkey2, (n_samples, 1))

            # OT conditional flow
            psi = t * x + (1.0 - (1.0 - sigma_min) * t) * x0

            # OT conditional vector field
            onemsig = 1.0 - sigma_min
            ucond = (x - onemsig * x0) / (1.0 - onemsig * t)

            # Simple MSE loss
            loss = jnp.mean((psi - ucond) ** 2)
            return loss

        mean_time, std_time = timeit_jax(compute_cfm_loss, x, key)

        return {
            "cfm_loss_mean_ms": mean_time,
            "cfm_loss_std_ms": std_time,
        }


class SamplerBenchmarks:
    """Benchmarks for sampler operations."""

    @staticmethod
    def bench_metropolis_acceptance(n_samples: int) -> Dict[str, float]:
        """Benchmark Metropolis acceptance criterion."""
        from rmc.modules.sampler import accept_Metropolis

        key = jax.random.PRNGKey(42)
        E_old = jax.random.normal(key, (n_samples,))
        E_new = jax.random.normal(key, (n_samples,))

        mean_time, std_time = timeit_jax(accept_Metropolis, key, E_old, E_new)

        return {
            "metropolis_mean_ms": mean_time,
            "metropolis_std_ms": std_time,
        }

    @staticmethod
    def bench_energy_computation(n_samples: int, dim: int) -> Dict[str, float]:
        """Benchmark HMC energy computation."""
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (n_samples, dim))
        p_rn = jax.random.normal(key, (n_samples, dim))

        @jax.jit
        def compute_energy(q, p_rn):
            """Compute total energy (potential + kinetic)."""
            # Simple potential energy (Gaussian)
            potentialE = jnp.sum(q**2, axis=1) / 2.0
            # Kinetic energy
            kineticE = jnp.sum(p_rn**2, axis=1) / 2.0
            return potentialE + kineticE

        mean_time, std_time = timeit_jax(compute_energy, q, p_rn)

        return {
            "energy_mean_ms": mean_time,
            "energy_std_ms": std_time,
        }


class DensityBenchmarks:
    """Benchmarks for density evaluations."""

    @staticmethod
    def bench_log_likelihood(n_samples: int, dim: int, data_dim: int) -> Dict[str, float]:
        """Benchmark log-likelihood computation for linear regression."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (n_samples, dim))
        data_x = jax.random.normal(key, (data_dim, dim))
        data_y = jax.random.normal(key, (data_dim,))
        var = 1.0

        @jax.jit
        def log_likelihood(x, data_x, data_y, var):
            """Compute log-likelihood for linear regression."""
            y_eval = jnp.sum(x.reshape((-1, 1, x.shape[-1])) * data_x, axis=-1)
            ll = (
                -0.5 * jnp.sum((data_y - y_eval) ** 2 / var, axis=-1)
                - 0.5 * data_dim * jnp.log(2 * jnp.pi)
                - 0.5 * data_dim * jnp.log(var)
            )
            return ll.squeeze()

        mean_time, std_time = timeit_jax(log_likelihood, x, data_x, data_y, var)

        return {
            "log_likelihood_mean_ms": mean_time,
            "log_likelihood_std_ms": std_time,
        }


def run_benchmark_suite(device: str = "cpu") -> Dict[str, Dict]:
    """Run complete benchmark suite.

    Args:
        device: Device to run on ("cpu", "gpu", or "both")

    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Running RMC Performance Benchmarks on {device.upper()}")
    print(f"{'='*70}\n")

    # Configure device
    if device == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
    elif device == "gpu":
        jax.config.update('jax_platform_name', 'gpu')

    # Print device info
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    results = {}

    # Test configurations (small, medium, large)
    configs = [
        {"name": "small", "n_samples": 100, "dim": 10, "data_dim": 50},
        {"name": "medium", "n_samples": 1000, "dim": 50, "data_dim": 100},
        {"name": "large", "n_samples": 10000, "dim": 100, "data_dim": 500},
    ]

    for config in configs:
        print(f"\n{'-'*70}")
        print(f"Configuration: {config['name'].upper()}")
        print(f"  n_samples: {config['n_samples']}, dim: {config['dim']}, data_dim: {config['data_dim']}")
        print(f"{'-'*70}")

        config_results = {}

        # Kernel benchmarks
        print("\n[1/6] Running kernel benchmarks...")
        config_results["kernels"] = {
            **KernelBenchmarks.bench_cdist(config["n_samples"], config["dim"]),
            **KernelBenchmarks.bench_rbf_gramm(config["n_samples"], config["dim"]),
        }
        print(f"  cdist: {config_results['kernels']['cdist_mean_ms']:.3f} ± {config_results['kernels']['cdist_std_ms']:.3f} ms")
        print(f"  RBF Gramm: {config_results['kernels']['rbf_gramm_mean_ms']:.3f} ± {config_results['kernels']['rbf_gramm_std_ms']:.3f} ms")

        # SVGD benchmarks
        print("\n[2/6] Running SVGD benchmarks...")
        config_results["svgd"] = SVGDBenchmarks.bench_svgd_kernel(config["n_samples"], config["dim"])
        print(f"  SVGD kernel: {config_results['svgd']['svgd_kernel_mean_ms']:.3f} ± {config_results['svgd']['svgd_kernel_std_ms']:.3f} ms")

        # CFM benchmarks
        print("\n[3/6] Running CFM benchmarks...")
        config_results["cfm"] = {
            **CFMBenchmarks.bench_cfm_ucond(config["n_samples"], config["dim"]),
            **CFMBenchmarks.bench_cfm_loss(config["n_samples"], config["dim"]),
        }
        print(f"  CFM ucond: {config_results['cfm']['cfm_ucond_mean_ms']:.3f} ± {config_results['cfm']['cfm_ucond_std_ms']:.3f} ms")
        print(f"  CFM loss: {config_results['cfm']['cfm_loss_mean_ms']:.3f} ± {config_results['cfm']['cfm_loss_std_ms']:.3f} ms")

        # Sampler benchmarks
        print("\n[4/6] Running sampler benchmarks...")
        config_results["sampler"] = {
            **SamplerBenchmarks.bench_metropolis_acceptance(config["n_samples"]),
            **SamplerBenchmarks.bench_energy_computation(config["n_samples"], config["dim"]),
        }
        print(f"  Metropolis: {config_results['sampler']['metropolis_mean_ms']:.3f} ± {config_results['sampler']['metropolis_std_ms']:.3f} ms")
        print(f"  Energy: {config_results['sampler']['energy_mean_ms']:.3f} ± {config_results['sampler']['energy_std_ms']:.3f} ms")

        # Density benchmarks
        print("\n[5/6] Running density benchmarks...")
        config_results["density"] = DensityBenchmarks.bench_log_likelihood(
            config["n_samples"], config["dim"], config["data_dim"]
        )
        print(f"  Log-likelihood: {config_results['density']['log_likelihood_mean_ms']:.3f} ± {config_results['density']['log_likelihood_std_ms']:.3f} ms")

        results[config["name"]] = config_results

    print(f"\n{'='*70}")
    print("Benchmark suite completed!")
    print(f"{'='*70}\n")

    return results


def compare_cpu_gpu():
    """Compare CPU vs GPU performance."""
    print("\n" + "="*70)
    print("CPU vs GPU Performance Comparison")
    print("="*70)

    # Run on CPU
    print("\nRunning on CPU...")
    cpu_results = run_benchmark_suite("cpu")

    # Check if GPU is available
    try:
        jax.config.update('jax_platform_name', 'gpu')
        gpu_devices = jax.devices('gpu')
        if len(gpu_devices) > 0:
            print("\nRunning on GPU...")
            gpu_results = run_benchmark_suite("gpu")

            # Print comparison
            print("\n" + "="*70)
            print("Speedup (CPU time / GPU time)")
            print("="*70)

            for config in ["small", "medium", "large"]:
                print(f"\n{config.upper()} configuration:")
                cpu_cfg = cpu_results[config]
                gpu_cfg = gpu_results[config]

                # Compare kernels
                rbf_speedup = cpu_cfg["kernels"]["rbf_gramm_mean_ms"] / gpu_cfg["kernels"]["rbf_gramm_mean_ms"]
                print(f"  RBF Gramm: {rbf_speedup:.2f}x")

                # Compare SVGD
                svgd_speedup = cpu_cfg["svgd"]["svgd_kernel_mean_ms"] / gpu_cfg["svgd"]["svgd_kernel_mean_ms"]
                print(f"  SVGD kernel: {svgd_speedup:.2f}x")

                # Compare CFM
                cfm_speedup = cpu_cfg["cfm"]["cfm_loss_mean_ms"] / gpu_cfg["cfm"]["cfm_loss_mean_ms"]
                print(f"  CFM loss: {cfm_speedup:.2f}x")
        else:
            print("\nGPU not available. Skipping GPU benchmarks.")
    except Exception as e:
        print(f"\nGPU not available: {e}")
        print("Skipping GPU benchmarks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMC Performance Benchmarks")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "both"],
        help="Device to run benchmarks on"
    )

    args = parser.parse_args()

    if args.device == "both":
        compare_cpu_gpu()
    else:
        run_benchmark_suite(args.device)

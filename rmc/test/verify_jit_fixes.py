#!/usr/bin/env python3
"""Comprehensive verification script for JIT optimizations.

This script runs actual sampling/training workflows to verify that
JIT optimizations don't break stateful operations.
"""

import sys
import traceback
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


class VerificationResult:
    """Store verification test results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = []
        self.failed = []
        self.errors = []

    def add_pass(self, test: str):
        self.passed.append(test)

    def add_fail(self, test: str, message: str):
        self.failed.append((test, message))

    def add_error(self, test: str, error: str):
        self.errors.append((test, error))

    def print_summary(self):
        total = len(self.passed) + len(self.failed) + len(self.errors)
        print(f"\n{'='*70}")
        print(f"{self.name}")
        print(f"{'='*70}")
        print(f"✅ Passed: {len(self.passed)}/{total}")
        if self.failed:
            print(f"❌ Failed: {len(self.failed)}/{total}")
            for test, msg in self.failed:
                print(f"  - {test}: {msg}")
        if self.errors:
            print(f"💥 Errors: {len(self.errors)}/{total}")
            for test, err in self.errors:
                print(f"  - {test}: {err}")
        print()

        return len(self.failed) == 0 and len(self.errors) == 0


def verify_hmc_tempering():
    """Verify HMC tempering updates are visible."""
    result = VerificationResult("HMC Tempering Verification")

    try:
        from rmc.modules.sampler import HMC
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import BaseLogDensity
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        class SimpleGaussian(BaseLogDensity):
            def __init__(self):
                pass

            def log_target(self, x):
                return -0.5 * jnp.sum(x**2, axis=-1)

        config = ConfigDict()
        config["sample_shape"] = (10, 5)
        config["density_cl"] = SimpleGaussian()
        config["step_size"] = 0.01
        config["numleapfrog"] = 5
        config["seed"] = 42
        config["log_freq"] = 100
        config["maxiter"] = 5

        mu = jnp.zeros((1, 5))
        cov = jnp.eye(5).reshape((1, 5, 5))
        config["initial_sampler_cl"] = PackedMultivariateNormal(mu, cov)

        hmc = HMC(config)

        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (10, 5))
        p_rn = jax.random.normal(key, (10, 5))

        # Test 1: Tempering changes visible
        hmc.tempering = 0.5
        energy_1 = hmc.compute_energy(q, p_rn)

        hmc.tempering = 1.0
        energy_2 = hmc.compute_energy(q, p_rn)

        if jnp.allclose(energy_1, energy_2):
            result.add_fail(
                "Tempering update visibility",
                "Energy doesn't change with tempering - JIT may be capturing state"
            )
        else:
            result.add_pass("Tempering update visibility")

        # Test 2: Full sampling loop
        config["maxiter"] = 10
        config["tempering_fn"] = lambda t: 0.1 + 0.9 * (t / 10.0)
        hmc = HMC(config)

        samples = hmc.sample()

        if samples.shape == (10, 5):
            result.add_pass("Full HMC sampling loop completes")
        else:
            result.add_fail("Full HMC sampling loop", f"Wrong shape: {samples.shape}")

        if jnp.all(jnp.isfinite(samples)):
            result.add_pass("HMC samples are finite")
        else:
            result.add_fail("HMC samples", "Contains NaN or Inf")

    except Exception as e:
        result.add_error("HMC verification", str(e))
        traceback.print_exc()

    return result


def verify_smc_ess():
    """Verify SMC ESS computation with updated weights."""
    result = VerificationResult("SMC ESS Verification")

    try:
        from rmc.modules.sampler import SMC
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import LogDensityPath
        from rmc.utils.packed_distributions import PackedMultivariateNormal


        class SimplePath(LogDensityPath):
            def __init__(self):
                self.d = 3
                self.initial = PackedMultivariateNormal(
                    jnp.zeros((1, 3)),
                    jnp.eye(3).reshape((1, 3, 3))
                )

            def log_initial(self, x):
                return self.initial.log_pdf(x)

            def log_target(self, x, t):
                # Simple interpolation
                return -0.5 * jnp.sum(x**2, axis=-1) * t


        config = ConfigDict()
        config["density_cl"] = SimplePath()
        config["step_size"] = 0.01
        config["numsteps"] = 5
        config["seed"] = 42
        config["log_freq"] = 100
        config["maxiter"] = 5
        config["sample_shape"] = (20, 3)
        config["tempering_fn"] = lambda t: t / 5.0
        config["ESS_thres"] = 0.5

        smc = SMC(20, 5, config)

        # Test ESS computation
        smc.logw = jnp.zeros(20)
        ess_uniform = smc.compute_ess()

        if jnp.isclose(ess_uniform, 1.0, rtol=1e-3):
            result.add_pass("ESS = 1.0 for uniform weights")
        else:
            result.add_fail("ESS uniform weights", f"Got {ess_uniform}, expected ~1.0")

        # Update weights
        smc.logw = jax.random.normal(jax.random.PRNGKey(42), (20,))
        ess_nonuniform = smc.compute_ess()

        if ess_nonuniform < ess_uniform:
            result.add_pass("ESS decreases with non-uniform weights")
        else:
            result.add_fail(
                "ESS weight update visibility",
                f"ESS didn't decrease: {ess_uniform} -> {ess_nonuniform}"
            )

    except Exception as e:
        result.add_error("SMC ESS verification", str(e))
        traceback.print_exc()

    return result


def verify_svgd_adaptation():
    """Verify SVGD kernel parameter adaptation."""
    result = VerificationResult("SVGD Kernel Adaptation Verification")

    try:
        from rmc.modules.svgd import SVGD, svgd_kernel_pure
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import BaseLogDensity

        class SimpleGaussian(BaseLogDensity):
            def __init__(self):
                pass

            def log_target(self, x):
                return -0.5 * jnp.sum(x**2, axis=-1)

            def der_log_target_proposal(self, x, tempering=None):
                return -x

        # Test pure function
        key = jax.random.PRNGKey(42)
        particles_1 = jax.random.normal(key, (30, 5)) * 0.5
        particles_2 = jax.random.normal(jax.random.PRNGKey(43), (30, 5)) * 2.0

        Kxy_1, dxkxy_1, kpar_1 = svgd_kernel_pure(particles_1, -1.0, -1.0)
        Kxy_2, dxkxy_2, kpar_2 = svgd_kernel_pure(particles_2, -1.0, -1.0)

        if not jnp.isclose(kpar_1, kpar_2, rtol=1e-3):
            result.add_pass("Adaptive kernel parameter changes with data")
        else:
            result.add_fail(
                "Kernel adaptation",
                f"Parameter didn't adapt: {kpar_1} vs {kpar_2}"
            )

        # Test determinism
        Kxy_1b, dxkxy_1b, kpar_1b = svgd_kernel_pure(particles_1, -1.0, -1.0)

        if jnp.allclose(Kxy_1, Kxy_1b) and jnp.allclose(dxkxy_1, dxkxy_1b):
            result.add_pass("SVGD kernel pure function is deterministic")
        else:
            result.add_fail("SVGD determinism", "Pure function not deterministic")

        # Test full SVGD loop
        config = ConfigDict()
        config["maxiter"] = 20
        config["step_size"] = 0.01
        config["density_cl"] = SimpleGaussian()
        config["kernel_parameter"] = 1.0
        config["update_weight"] = 0.9
        config["seed"] = 42
        config["sample_shape"] = (40, 5)
        config["log_freq"] = 100
        config["initial_sampler_cl"] = lambda key, shape: jax.random.normal(key, shape + (5,))

        svgd = SVGD(40, config)
        samples = svgd.sample()

        if samples.shape == (40, 5):
            result.add_pass("Full SVGD sampling loop completes")
        else:
            result.add_fail("SVGD sampling", f"Wrong shape: {samples.shape}")

        if jnp.all(jnp.isfinite(samples)):
            result.add_pass("SVGD samples are finite")
        else:
            result.add_fail("SVGD samples", "Contains NaN or Inf")

    except Exception as e:
        result.add_error("SVGD verification", str(e))
        traceback.print_exc()

    return result


def verify_cfm_constants():
    """Verify CFM only accesses constant attributes."""
    result = VerificationResult("CFM Constants Verification")

    try:
        from rmc.modules.cfm import ConditionalFlowMatching
        from rmc.flax.nn_config_dict import NNConfigDict

        config = NNConfigDict()
        config["dim"] = 10
        config["sigma_min"] = 0.001
        config["nn_type"] = "time_embed"
        config["layer_widths"] = [32, 32]
        config["activation_func"] = "gelu"
        config["seed"] = 42

        key = jax.random.PRNGKey(42)
        samples = jax.random.normal(key, (50, 10))

        class DummyDensity:
            pass

        cfm = ConditionalFlowMatching(config, DummyDensity(), samples, model_type="ot")

        x1 = jax.random.normal(key, (10, 10))
        x = jax.random.normal(key, (10, 10))
        t = jax.random.uniform(key, (10, 1))

        # Test determinism
        result_1 = cfm.eval_ucond_ot(x1, x, t)
        result_2 = cfm.eval_ucond_ot(x1, x, t)

        if jnp.allclose(result_1, result_2):
            result.add_pass("CFM eval_ucond_ot is deterministic")
        else:
            result.add_fail("CFM determinism", "Function not deterministic")

        # Test other methods
        psi = cfm.eval_psi(x1, x, t)
        if psi.shape == (10, 10):
            result.add_pass("CFM eval_psi works")
        else:
            result.add_fail("CFM eval_psi", f"Wrong shape: {psi.shape}")

        ucond = cfm.eval_ucond(x1, x, t)
        if ucond.shape == (10, 10):
            result.add_pass("CFM eval_ucond works")
        else:
            result.add_fail("CFM eval_ucond", f"Wrong shape: {ucond.shape}")

    except Exception as e:
        result.add_error("CFM verification", str(e))
        traceback.print_exc()

    return result


def verify_density_immutability():
    """Verify density functions handle data correctly."""
    result = VerificationResult("Density Data Immutability Verification")

    try:
        from rmc.utils.density import LinearRegressionDensity
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        dim = 5
        prior = PackedMultivariateNormal(
            jnp.zeros((1, dim)),
            jnp.eye(dim).reshape((1, dim, dim))
        )

        key = jax.random.PRNGKey(42)
        data_x = jax.random.normal(key, (20, dim))
        data_y = jax.random.normal(key, (20,))
        prior_mean_vec = 1.0 * jnp.ones((1, dim))
        prior_std_vec = 10.0 * jnp.ones((1, dim))
        noise_std = 0.1

        density = LinearRegressionDensity(
                    dim,
                    data_x,
                    data_y,
                    noise_std,
                    prior_mean_vec,
                    prior_std_vec,
        )
        x = jax.random.normal(key, (10, dim))

        # Test determinism
        ll_1 = density.log_likelihood(x)
        ll_2 = density.log_likelihood(x)

        if jnp.allclose(ll_1, ll_2):
            result.add_pass("Density log_likelihood is deterministic")
        else:
            result.add_fail("Density determinism", "log_likelihood not deterministic")

        # Verify data hasn't changed
        if jnp.allclose(density.data_x, data_x):
            result.add_pass("Density data_x unchanged")
        else:
            result.add_fail("Density data", "data_x was modified")

        if jnp.allclose(density.data_y, data_y):
            result.add_pass("Density data_y unchanged")
        else:
            result.add_fail("Density data", "data_y was modified")

    except Exception as e:
        result.add_error("Density verification", str(e))
        traceback.print_exc()

    return result


def verify_vectorization():
    """Verify vectorization produces correct results."""
    result = VerificationResult("Vectorization Correctness Verification")

    try:
        from rmc.modules.svgd import svgd_kernel_pure
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (30, 5))
        kpar = 1.5

        # Compute with vectorized version
        Kxy, dxkxy_vec, _ = svgd_kernel_pure(particles, kpar, kpar)

        # Compute with loop (reference)
        dxkxy_loop = -jnp.matmul(Kxy, particles)
        sumkxy = jnp.sum(Kxy, axis=1)
        for i in range(particles.shape[1]):
            dxkxy_loop = dxkxy_loop.at[:, i].add(jnp.multiply(particles[:, i], sumkxy))
        dxkxy_loop = dxkxy_loop / kpar

        if jnp.allclose(dxkxy_vec, dxkxy_loop, rtol=1e-5):
            result.add_pass("Vectorized SVGD kernel matches loop version")
        else:
            max_diff = jnp.max(jnp.abs(dxkxy_vec - dxkxy_loop))
            result.add_fail(
                "Vectorization correctness",
                f"Max difference: {max_diff}"
            )

    except Exception as e:
        result.add_error("Vectorization verification", str(e))
        traceback.print_exc()

    return result


def main():
    """Run all verification tests."""
    print("="*70)
    print("RMC JIT Optimization Verification Suite")
    print("="*70)
    print("\nRunning comprehensive tests to verify JIT optimizations...")
    print("This will test actual sampling workflows to catch state bugs.\n")

    # Run all verifications
    results = []

    print("[1/7] Verifying HMC tempering...")
    results.append(verify_hmc_tempering())

    print("[2/7] Verifying SMC ESS computation...")
    results.append(verify_smc_ess())

    print("[3/7] Verifying SVGD kernel adaptation...")
    results.append(verify_svgd_adaptation())

    print("[4/7] Verifying CFM constants...")
    results.append(verify_cfm_constants())

    print("[5/7] Verifying density data immutability...")
    results.append(verify_density_immutability())

    print("[6/7] Verifying vectorization correctness...")
    results.append(verify_vectorization())

    # Print summaries
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)

    all_passed = True
    for result in results:
        if not result.print_summary():
            all_passed = False

    # Overall summary
    total_passed = sum(len(r.passed) for r in results)
    total_failed = sum(len(r.failed) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    total_tests = total_passed + total_failed + total_errors

    print("="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed:   {total_passed} ({100*total_passed/total_tests:.1f}%)")
    print(f"❌ Failed:   {total_failed} ({100*total_failed/total_tests:.1f}%)")
    print(f"💥 Errors:   {total_errors} ({100*total_errors/total_tests:.1f}%)")
    print()

    if all_passed:
        print("✅ All verifications PASSED! JIT optimizations are working correctly.")
        print()
        return 0
    else:
        print("⚠️  Some verifications FAILED. Check results above.")
        print()
        print("Common issues:")
        print("  - If tempering/ESS tests fail: JIT captured mutable state")
        print("  - If adaptation tests fail: Parameters not updating")
        print("  - If sampling loops fail: Check for runtime errors")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

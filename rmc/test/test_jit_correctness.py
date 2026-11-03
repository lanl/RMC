"""Comprehensive tests to verify JIT optimizations don't break stateful operations.

This test suite specifically checks for bugs where @jax.jit captures mutable
instance state at compilation time, causing state updates to be invisible.
"""

from jax._src.random import RealArray

import pytest
import jax
import jax.numpy as jnp
import numpy as np


class TestHMCStatefulness:
    """Test that HMC sampler correctly handles state updates."""

    def test_tempering_updates_visible(self):
        """Test that changes to tempering are visible to compute_energy."""
        from rmc.modules.sampler import HMC
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import BaseLogDensity
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        # Create simple Gaussian density
        class SimpleGaussian(BaseLogDensity):
            def __init__(self):
                pass

            def log_target(self, x):
                return -0.5 * jnp.sum(x**2, axis=-1)

        # Create HMC config
        config = ConfigDict()
        config["sample_shape"] = (10, 5)
        config["density_cl"] = SimpleGaussian()
        config["step_size"] = 0.01
        config["numsteps"] = 10
        config["seed"] = 42
        config["log_freq"] = 100
        config["maxiter"] = 5

        # Initial sampler from Gaussian
        mu = jnp.zeros((1, 5))
        cov = jnp.eye(5).reshape((1, 5, 5))
        config["initial_sampler_cl"] = PackedMultivariateNormal(mu, cov)

        hmc = HMC(config)

        # Initialize
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (10, 5))
        p_rn = jax.random.normal(key, (10, 5))

        # Test 1: Different tempering gives different energy
        hmc.tempering = 0.5
        energy_1 = hmc.compute_energy(q, p_rn)

        hmc.tempering = 1.0
        energy_2 = hmc.compute_energy(q, p_rn)

        # Should be different (unless tempering=None, but we set it explicitly)
        assert not jnp.allclose(energy_1, energy_2), \
            "Energy should change with tempering! JIT may have captured initial value."

        # Test 2: Tempering = None vs explicit value
        hmc.tempering = None
        energy_none = hmc.compute_energy(q, p_rn)

        hmc.tempering = 1.0
        energy_one = hmc.compute_energy(q, p_rn)

        # These should be equal (tempering=1 is identity)
        assert jnp.allclose(energy_none, energy_one, rtol=1e-5), \
            "Energy with tempering=1.0 should equal tempering=None"

    def test_tempering_function_updates(self):
        """Test that tempering function is called each iteration."""
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
        config["sample_shape"] = (5, 3)
        config["density_cl"] = SimpleGaussian()
        config["step_size"] = 0.01
        config["numsteps"] = 5
        config["seed"] = 42
        config["log_freq"] = 100
        config["maxiter"] = 10

        mu = jnp.zeros((1, 3))
        cov = jnp.eye(3).reshape((1, 3, 3))
        config["initial_sampler_cl"] = PackedMultivariateNormal(mu, cov)

        # Linear tempering schedule
        config["tempering_fn"] = lambda t: 0.1 + 0.9 * (t / 10.0)

        hmc = HMC(config)

        # Manually test a few iterations
        tempering_values = []
        for i in range(5):
            hmc.itnum = i
            if hmc.tempering_fn is not None:
                hmc.tempering = hmc.tempering_fn(hmc.itnum)
            tempering_values.append(hmc.tempering)

        # Should increase linearly
        assert len(set(tempering_values)) == 5, \
            "Tempering should change each iteration"
        assert tempering_values == sorted(tempering_values), \
            "Tempering should increase monotonically"


class TestSMCStatefulness:
    """Test that SMC sampler correctly handles state updates."""

    def test_ess_recomputation(self):
        """Test that ESS is recomputed with updated log weights."""
        from rmc.modules.sampler import SMC
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import LogDensityPath
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        # Create simple path from Gaussian to Gaussian
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
                return self.log_initial(x) * t

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

        # Initialize log weights
        smc.logw = jnp.zeros(20)

        # Compute ESS with uniform weights
        ess_uniform = smc.compute_ess()
        assert jnp.isclose(ess_uniform, 1.0, rtol=1e-5), \
            "ESS should be 1.0 for uniform weights"

        # Update weights to non-uniform
        smc.logw = jax.random.normal(jax.random.PRNGKey(42), (20,))

        # Compute ESS with non-uniform weights
        ess_nonuniform = smc.compute_ess()

        # Should be different and less than 1.0
        assert ess_nonuniform < ess_uniform, \
            "ESS should decrease with non-uniform weights! JIT may have captured initial weights."
        assert ess_nonuniform < 1.0, \
            "ESS with non-uniform weights should be < 1.0"


class TestSVGDStatefulness:
    """Test that SVGD correctly handles adaptive parameters."""

    def test_kernel_parameter_adaptation(self):
        """Test that kernel parameter adapts when kpar0 < 0."""
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

        config = ConfigDict()
        config["maxiter"] = 10
        config["step_size"] = 0.01
        config["density_cl"] = SimpleGaussian()
        config["kernel_parameter"] = -1  # Adaptive
        config["update_weight"] = 0.9
        config["seed"] = 42
        config["sample_shape"] = (50, 10)
        config["log_freq"] = 100
        config["initial_sampler_cl"] = lambda key, shape: jax.random.normal(key, shape + (10,))

        svgd = SVGD(50, config)

        # Get initial kernel parameter
        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (50, 10))

        # Call svgd_kernel multiple times with different particles
        particles_1 = jax.random.normal(key, (50, 10)) * 0.5
        kpar_1 = -1.
        Kxy_1, dxkxy_1, kpar_1 = svgd_kernel_pure(particles_1, kpar_1, kpar0 = -1.)

        particles_2 = jax.random.normal(jax.random.PRNGKey(43), (50, 10)) * 2.0
        kpar_2 = -1.
        Kxy_2, dxkxy_2, kpar_2 = svgd_kernel_pure(particles_2, kpar_2, kpar0 = -1.)

        # Kernel parameters should be different (particles have different scales)
        assert not jnp.isclose(kpar_1, kpar_2, rtol=1e-3), \
            "Adaptive kernel parameter should change with different particle distributions"

    def test_svgd_kernel_pure_function(self):
        """Test that svgd_kernel_pure is a pure function (deterministic)."""
        from rmc.modules.svgd import svgd_kernel_pure

        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (30, 5))
        kpar = 1.0
        kpar0 = 1.0  # Don't adapt

        # Call twice with same inputs
        Kxy_1, dxkxy_1, kpar_out_1 = svgd_kernel_pure(particles, kpar, kpar0)
        Kxy_2, dxkxy_2, kpar_out_2 = svgd_kernel_pure(particles, kpar, kpar0)

        # Should be identical
        assert jnp.allclose(Kxy_1, Kxy_2), "Pure function should be deterministic"
        assert jnp.allclose(dxkxy_1, dxkxy_2), "Pure function should be deterministic"
        assert kpar_out_1 == kpar_out_2, "Pure function should be deterministic"

    def test_iteration_counter_visibility(self):
        """Test that iteration counter is properly tracked."""
        from rmc.modules.svgd import SVGD
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import BaseLogDensity

        class SimpleGaussian(BaseLogDensity):
            def __init__(self):
                pass

            def log_target(self, x):
                return -0.5 * jnp.sum(x**2, axis=-1)

            def der_log_target_proposal(self, x, tempering=None):
                return -x

        config = ConfigDict()
        config["maxiter"] = 5
        config["step_size"] = 0.01
        config["density_cl"] = SimpleGaussian()
        config["kernel_parameter"] = 1.0
        config["update_weight"] = 0.9
        config["seed"] = 42
        config["sample_shape"] = (20, 5)
        config["log_freq"] = 100
        config["initial_sampler_cl"] = lambda key, shape: jax.random.normal(key, shape + (5,))

        svgd = SVGD(20, config)

        # Manually step through iterations
        key = jax.random.PRNGKey(42)
        samples = jax.random.normal(key, (20, 5))

        iterations = []
        for i in range(3):
            svgd.itnum = i
            iterations.append(svgd.itnum)
            key, samples = svgd.step(key, samples)

        # Should track correctly
        assert iterations == [0, 1, 2], "Iteration counter should increment"


class TestCFMStatefulness:
    """Test that CFM correctly handles constant attributes."""

    def test_cfm_constants_only(self):
        """Test that CFM only accesses constants, not mutable state."""
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

        # Test that sigma_min is constant
        original_sigma_min = cfm.sigma_min

        x1 = jax.random.normal(key, (10, 10))
        x = jax.random.normal(key, (10, 10))
        t = jax.random.uniform(key, (10, 1))

        result_1 = cfm.eval_ucond_ot(x1, x, t)

        # Verify sigma_min hasn't changed
        assert cfm.sigma_min == original_sigma_min, \
            "sigma_min should not change"

        # Call again
        result_2 = cfm.eval_ucond_ot(x1, x, t)

        # Results should be identical (deterministic)
        assert jnp.allclose(result_1, result_2), \
            "JIT compiled function should be deterministic with constants"

    def test_cfm_ucond_variations(self):
        """Test different CFM models (ve, ot) work correctly."""
        from rmc.modules.cfm import ConditionalFlowMatching
        from rmc.flax.nn_config_dict import NNConfigDict
        from rmc.utils.schedule import CosineSchedule

        key = jax.random.PRNGKey(42)
        samples = jax.random.normal(key, (30, 5))

        class DummyDensity:
            pass

        # Test OT model
        config_ot = NNConfigDict()
        config_ot["dim"] = 5
        config_ot["sigma_min"] = 0.001
        config_ot["nn_type"] = "time_embed"
        config_ot["layer_widths"] = [16, 16]
        config_ot["activation_func"] = "gelu"
        config_ot["seed"] = 42

        cfm_ot = ConditionalFlowMatching(config_ot, DummyDensity(), samples, model_type="ot")

        x1 = jax.random.normal(key, (10, 5))
        x = jax.random.normal(key, (10, 5))
        t = jax.random.uniform(key, (10, 1), minval=0.1, maxval=0.9)

        ucond_ot = cfm_ot.eval_ucond_ot(x1, x, t)
        assert ucond_ot.shape == (10, 5), "OT ucond shape incorrect"

        # Test VE model
        config_ve = NNConfigDict()
        config_ve["dim"] = 5
        config_ve["nn_type"] = "time_embed"
        config_ve["layer_widths"] = [16, 16]
        config_ve["activation_func"] = "gelu"
        config_ve["seed"] = 42

        schedule = CosineSchedule()
        cfm_ve = ConditionalFlowMatching(
            config_ve, DummyDensity(), samples,
            model_type="ve", schedule=schedule
        )

        ucond_ve = cfm_ve.eval_ucond_ve(x1, x, t)
        assert ucond_ve.shape == (10, 5), "VE ucond shape incorrect"


class TestDensityStatefulness:
    """Test that density functions handle data correctly."""

    def test_density_data_immutability(self):
        """Test that density only accesses data set at initialization."""
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

        # Evaluate likelihood
        x = jax.random.normal(key, (10, dim))
        ll_1 = density.log_likelihood(x)

        # Data should not have changed
        assert jnp.allclose(density.data_x, data_x), "data_x should not change"
        assert jnp.allclose(density.data_y, data_y), "data_y should not change"

        # Call again - should be deterministic
        ll_2 = density.log_likelihood(x)
        assert jnp.allclose(ll_1, ll_2), "log_likelihood should be deterministic"


class TestVectorization:
    """Test that vectorization produces correct results."""

    def test_svgd_kernel_vectorization(self):
        """Test vectorized SVGD kernel matches reference implementation."""
        from rmc.modules.svgd import svgd_kernel_pure
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (30, 5))
        kpar = 1.5
        kpar0 = 1.5  # Don't adapt

        # Compute with vectorized version
        Kxy, dxkxy_vec, _ = svgd_kernel_pure(particles, kpar, kpar0)

        # Compute reference with loop (original implementation)
        dxkxy_loop = -jnp.matmul(Kxy, particles)
        sumkxy = jnp.sum(Kxy, axis=1)
        for i in range(particles.shape[1]):
            dxkxy_loop = dxkxy_loop.at[:, i].add(jnp.multiply(particles[:, i], sumkxy))
        dxkxy_loop = dxkxy_loop / kpar

        # Should match
        assert jnp.allclose(dxkxy_vec, dxkxy_loop, rtol=1e-5), \
            "Vectorized kernel derivative should match loop version"


class TestIntegrationWorkflows:
    """Integration tests that run full sampling workflows."""

    def test_hmc_full_sampling_loop(self):
        """Test that HMC can complete a full sampling loop."""
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
        config["sample_shape"] = (20, 3)
        config["density_cl"] = SimpleGaussian()
        config["step_size"] = 0.05
        config["numleapfrog"] = 5
        config["seed"] = 42
        config["log_freq"] = 100
        config["maxiter"] = 10

        mu = jnp.zeros((1, 3))
        cov = jnp.eye(3).reshape((1, 3, 3))
        config["initial_sampler_cl"] = PackedMultivariateNormal(mu, cov)

        hmc = HMC(config)

        # Run full sampling
        samples = hmc.sample()

        assert samples.shape == (20, 3), "Final samples shape incorrect"
        assert jnp.all(jnp.isfinite(samples)), "Samples contain NaN or Inf"

    def test_svgd_full_sampling_loop(self):
        """Test that SVGD can complete a full sampling loop."""
        from rmc.modules.svgd import SVGD
        from rmc.utils.config_dict import ConfigDict
        from rmc.utils.density import BaseLogDensity

        class SimpleGaussian(BaseLogDensity):
            def __init__(self):
                pass

            def log_target(self, x):
                return -0.5 * jnp.sum(x**2, axis=-1)

            def der_log_target_proposal(self, x, tempering=None):
                return -x

        config = ConfigDict()
        config["maxiter"] = 10
        config["step_size"] = 0.01
        config["density_cl"] = SimpleGaussian()
        config["kernel_parameter"] = 1.0
        config["update_weight"] = 0.9
        config["seed"] = 42
        config["sample_shape"] = (30, 5)
        config["log_freq"] = 100
        config["initial_sampler_cl"] = lambda key, shape: jax.random.normal(key, shape + (5,))

        svgd = SVGD(30, config)

        # Run full sampling
        samples = svgd.sample()

        assert samples.shape == (30, 5), "Final samples shape incorrect"
        assert jnp.all(jnp.isfinite(samples)), "Samples contain NaN or Inf"

        # Samples should have moved from initial (mean should be closer to 0)
        assert jnp.abs(jnp.mean(samples)) < 1.0, \
            "SVGD samples should concentrate around target mean (0)"


class TestJITPurenesss:
    """Test that JIT-compiled functions are actually pure."""

    def test_jit_functions_are_deterministic(self):
        """Test that all JIT functions give same output for same input."""
        from rmc.utils.kernels import cdist, cdistSQ
        from rmc.modules.sampler import accept_Metropolis

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (20, 5))
        y = jax.random.normal(key, (20, 5))

        # Test cdist
        d1 = cdist(x, y)
        d2 = cdist(x, y)
        assert jnp.array_equal(d1, d2), "cdist should be deterministic"

        # Test cdistSQ
        d1 = cdistSQ(x, y)
        d2 = cdistSQ(x, y)
        assert jnp.array_equal(d1, d2), "cdistSQ should be deterministic"

        # Test accept_Metropolis (with same key)
        E_old = jax.random.normal(key, (20,))
        E_new = jax.random.normal(key, (20,))

        # Note: accept_Metropolis uses randomness, so we test with fixed key
        acc1 = accept_Metropolis(key, E_old, E_new)
        acc2 = accept_Metropolis(key, E_old, E_new)
        assert jnp.array_equal(acc1, acc2), "accept_Metropolis should be deterministic with fixed key"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

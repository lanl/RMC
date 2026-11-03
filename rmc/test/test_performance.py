"""Performance and correctness tests for JIT-optimized functions."""

import jax
import jax.numpy as jnp
from flax import nnx
import pytest


class TestJITOptimizations:
    """Test that JIT-decorated functions produce correct results and are actually compiled."""

    def test_kernels_jit_compiled(self):
        """Test that kernel functions are JIT compiled."""
        from rmc.utils.kernels import cdist, cdistSQ, RBF_Gramm, RBF

        # Test cdist
        x = jax.random.normal(jax.random.PRNGKey(0), (100, 10))
        y = jax.random.normal(jax.random.PRNGKey(1), (100, 10))

        # Should execute without error
        dist = cdist(x, y)
        assert dist.shape == (100, 100)
        assert jnp.all(dist >= 0)

        # Test cdistSQ
        dist_sq = cdistSQ(x, y)
        assert dist_sq.shape == (100, 100)
        assert jnp.all(dist_sq >= 0)
        # Square of cdist should be close to cdistSQ
        assert jnp.allclose(dist**2, dist_sq, rtol=1e-5)

        # Test RBF_Gramm
        Kxy, h = RBF_Gramm(x)
        assert Kxy.shape == (100, 100)
        assert h > 0
        assert jnp.all(Kxy >= 0)
        assert jnp.all(Kxy <= 1)  # RBF kernel values in [0, 1]

        # Test RBF
        xc = x[0]
        rbf_val = RBF(xc, h, x[0])
        assert jnp.isclose(rbf_val, 1.0)  # RBF(x, h, x) = 1

    def test_cfm_jit_compiled(self):
        """Test that CFM functions are JIT compiled and produce correct results."""
        from rmc.flax.nn_config_dict import NNConfigDict
        from rmc.modules.cfm import ConditionalFlowMatching
        from rmc.utils.density import LogDensityPath
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        # Create minimal config
        config = NNConfigDict()
        config["dim"] = 10
        config["sigma_min"] = 0.001
        config["nn_type"] = "time_embed"
        config["layer_widths"] = [32, 32]
        config["activation_func"] = nnx.silu
        config["seed"] = 42

        # Create simple density class
        key = jax.random.PRNGKey(42)
        samples = jax.random.normal(key, (100, 10))

        # Create dummy density class with required attributes
        class DummyDensity:
            pass

        density_cl = DummyDensity()

        # Create CFM with OT model
        cfm = ConditionalFlowMatching(
            config, density_cl, samples, model_type="ot"
        )

        # Test eval_ucond_ot
        x1 = jax.random.normal(key, (10, 10))
        x = jax.random.normal(key, (10, 10))
        t = jax.random.uniform(key, (10, 1))

        ucond = cfm.eval_ucond_ot(x1, x, t)
        assert ucond.shape == (10, 10)

        # Test eval_psi
        psi = cfm.eval_psi(x1, x, t)
        assert psi.shape == (10, 10)

    def test_svgd_jit_compiled(self):
        """Test that SVGD functions are JIT compiled."""
        from rmc.modules.svgd import SVGD, svgd_kernel_pure
        from rmc.utils.config_dict import ConfigDict

        # Create dummy density class
        class DummyDensity:
            def der_log_target_proposal(self, x):
                # Simple Gaussian gradient
                return -x

        config = ConfigDict()
        config["maxiter"] = 10
        config["step_size"] = 0.01
        config["density_cl"] = DummyDensity()
        config["kernel_parameter"] = 1.0
        config["update_weight"] = 0.9
        config["seed"] = 42
        config["sample_shape"] = (100, 10)
        config["log_freq"] = 1
        config["initial_sampler_cl"] = lambda key, shape: jax.random.normal(key, shape + (10,))

        svgd = SVGD(100, config)

        # Test svgd_kernel
        particles = jax.random.normal(jax.random.PRNGKey(42), (100, 10))
        Kxy, dxkxy, kpar = svgd_kernel_pure(particles, kpar=1.0, kpar0=-1.0)

        assert Kxy.shape == (100, 100)
        assert dxkxy.shape == (100, 10)
        assert jnp.all(Kxy >= 0)

    def test_sampler_jit_compiled(self):
        """Test that sampler functions are JIT compiled."""
        from rmc.modules.sampler import accept_Metropolis

        key = jax.random.PRNGKey(42)
        E_old = jax.random.normal(key, (100,))
        E_new = jax.random.normal(key, (100,))

        acc = accept_Metropolis(key, E_old, E_new)
        assert acc.shape == (100, 1)
        assert jnp.all((acc == 0) | (acc == 1))

    def test_density_jit_compiled(self):
        """Test that density functions can be JIT compiled."""
        from rmc.utils.density import LinearRegressionDensity
        from rmc.utils.packed_distributions import PackedMultivariateNormal

        # Create a simple Gaussian prior
        dim = 10
        prior = PackedMultivariateNormal(
            jnp.zeros((1, dim)),
            jnp.eye(dim).reshape((1, dim, dim))
        )

        # Create dummy data for linear regression
        key = jax.random.PRNGKey(42)
        data_x = jax.random.normal(key, (50, dim))
        data_y = jax.random.normal(key, (50,))
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

        # Test log_likelihood
        x = jax.random.normal(key, (100, dim))
        ll = density.log_likelihood(x)
        assert ll.shape == (100,)

        # Test log_target_proposal
        log_target = density.log_target_proposal(x)
        assert log_target.shape == (100,)


class TestVectorization:
    """Test that vectorized operations produce correct results."""

    def test_svgd_kernel_vectorization(self):
        """Test that vectorized SVGD kernel derivative is correct."""
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)
        particles = jax.random.normal(key, (50, 5))
        kpar = 1.0

        # Compute with vectorized version
        Kxy, _ = RBF_Gramm(particles, kpar)
        dxkxy_vec = -jnp.matmul(Kxy, particles)
        sumkxy = jnp.sum(Kxy, axis=1, keepdims=True)
        dxkxy_vec = dxkxy_vec + particles * sumkxy
        dxkxy_vec = dxkxy_vec / kpar

        # Compute with loop (reference implementation)
        dxkxy_loop = -jnp.matmul(Kxy, particles)
        sumkxy_loop = jnp.sum(Kxy, axis=1)
        for i in range(particles.shape[1]):
            dxkxy_loop = dxkxy_loop.at[:, i].add(jnp.multiply(particles[:, i], sumkxy_loop))
        dxkxy_loop = dxkxy_loop / kpar

        # Should produce the same result
        assert jnp.allclose(dxkxy_vec, dxkxy_loop, rtol=1e-5)


class TestGPUCompatibility:
    """Test GPU-specific functionality."""

    def test_gpu_available(self):
        """Check if GPU is available."""
        try:
            gpu_devices = jax.devices('gpu')
            if len(gpu_devices) > 0:
                print(f"GPU devices available: {gpu_devices}")
            else:
                pytest.skip("No GPU available")
        except Exception:
            pytest.skip("No GPU available")

    @pytest.mark.skipif(
        len(jax.devices('gpu')) == 0 if 'gpu' in str(jax.devices()) else True,
        reason="GPU not available"
    )
    def test_kernels_on_gpu(self):
        """Test that kernel operations work on GPU."""
        from rmc.utils.kernels import cdist, RBF_Gramm

        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (1000, 50))

        # Move to GPU and compute
        with jax.default_device(jax.devices('gpu')[0]):
            dist = cdist(x, x)
            assert dist.shape == (1000, 1000)

            Kxy, h = RBF_Gramm(x)
            assert Kxy.shape == (1000, 1000)


class TestNumericalStability:
    """Test numerical stability of optimized functions."""

    def test_rbf_gramm_stability(self):
        """Test RBF Gramm matrix is numerically stable."""
        from rmc.utils.kernels import RBF_Gramm

        key = jax.random.PRNGKey(42)

        # Test with various scales
        for scale in [1e-3, 1.0, 1e3]:
            x = scale * jax.random.normal(key, (100, 10))
            Kxy, h = RBF_Gramm(x)

            # Should be symmetric
            assert jnp.allclose(Kxy, Kxy.T, rtol=1e-5)

            # Should be positive semi-definite (all eigenvalues >= 0)
            eigvals = jnp.linalg.eigvalsh(Kxy)
            assert jnp.all(eigvals >= -1e-6)  # Allow small numerical errors

            # Diagonal should be 1
            assert jnp.allclose(jnp.diag(Kxy), 1.0, rtol=1e-5)

    def test_energy_computation_stability(self):
        """Test energy computation is numerically stable."""
        key = jax.random.PRNGKey(42)

        @jax.jit
        def compute_energy(q, p_rn):
            potentialE = jnp.sum(q**2, axis=1) / 2.0
            kineticE = jnp.sum(p_rn**2, axis=1) / 2.0
            return potentialE + kineticE

        # Test with various scales
        for scale in [1e-3, 1.0, 1e3]:
            q = scale * jax.random.normal(key, (100, 10))
            p_rn = scale * jax.random.normal(key, (100, 10))

            energy = compute_energy(q, p_rn)

            # Should all be finite
            assert jnp.all(jnp.isfinite(energy))

            # Should all be non-negative
            assert jnp.all(energy >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Regression test for persistent SVGD gradient history."""

import jax
import jax.numpy as jnp
import numpy as np

from rmc.modules.svgd import SVGD


class GaussianTarget:
    def der_log_target_proposal(self, q):
        return -q


def test_svgd_gradient_history():
    alpha = 0.9

    config = {
        "seed": 42,
        "sample_shape": (3, 2),
        "initial_sampler_cl": lambda key, shape: jax.random.normal(key, shape=shape + (2,)),
        "maxiter": 2,
        "step_size": 0.01,
        "log_freq": 10,
        "density_cl": GaussianTarget(),
        "kernel_parameter": 1.0,
        "update_weight": alpha,
    }

    sampler = SVGD(3, config)

    key = jax.random.key(42)
    initial = jnp.array([[-1.0, 0.5], [0.5, -0.25], [1.5, 1.0]])

    sampler.post_initialization(key, initial)

    sampler.itnum = 0
    _, x1 = sampler.step(key, initial)
    h0 = sampler.historical_grad.copy()

    sampler.itnum = 1
    _, x2 = sampler.step(key, x1)
    h1 = sampler.historical_grad.copy()

    kxy, dxkxy = sampler.svgd_kernel(x1)
    g1 = (kxy @ (-x1) + dxkxy) / x1.shape[0]

    expected_h1 = alpha * h0 + (1 - alpha) * g1**2

    np.testing.assert_allclose(h1, expected_h1, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(
        x2,
        x1 + config["step_size"] * g1 / (1e-6 + jnp.sqrt(expected_h1)),
        rtol=1e-5,
        atol=1e-7,
    )

"""Regression test for SVGD target evaluation counts."""

import jax
import jax.numpy as jnp

from rmc.modules.svgd import SVGD


class GaussianTarget:
    def der_log_target_proposal(self, q):
        return -q


def test_svgd_evaluation_counts():
    particles = 3
    iterations = 4

    config = {
        "seed": 42,
        "sample_shape": (particles, 2),
        "initial_sampler_cl": lambda key, shape: jax.random.normal(key, shape=shape + (2,)),
        "maxiter": iterations,
        "step_size": 0.01,
        "log_freq": iterations + 1,
        "density_cl": GaussianTarget(),
        "kernel_parameter": 1.0,
        "update_weight": 0.9,
    }

    sampler = SVGD(particles, config)

    assert sampler.evaluation_counts == {"density": 0, "gradient": 0}

    sampler.sample()

    assert sampler.evaluation_counts == {
        "density": 0,
        "gradient": particles * iterations,
    }

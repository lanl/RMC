"""Regression tests for HMC target evaluation counts."""

import jax
import jax.numpy as jnp

from rmc.modules.sampler import HMC


class GaussianTarget:
    def log_target_proposal(self, q, tempering):
        return -0.5 * jnp.sum(q**2, axis=1)

    def der_log_target_proposal(self, q, tempering):
        return -q


def test_hmc_evaluation_counts():
    chains = 2
    iterations = 3
    leapfrog_steps = 4

    config = {
        "seed": 42,
        "sample_shape": (chains, 2),
        "initial_sampler_cl": lambda key, shape: jax.random.normal(key, shape=shape + (2,)),
        "maxiter": iterations,
        "numleapfrog": leapfrog_steps,
        "step_size": 0.01,
        "log_freq": iterations + 1,
        "density_cl": GaussianTarget(),
    }

    sampler = HMC(config)

    assert sampler.evaluation_counts == {"density": 0, "gradient": 0}

    sampler.sample()

    assert sampler.evaluation_counts == {
        "density": 2 * chains * iterations,
        "gradient": (leapfrog_steps + 1) * chains * iterations,
    }

"""Regression test for SMC target evaluation counts."""

import jax
import jax.numpy as jnp

from rmc.modules.sampler import SMC


class GaussianTarget:
    def log_target_proposal(self, q, tempering):
        return -0.5 * jnp.sum(q**2, axis=1)

    def der_log_target_proposal(self, q, tempering):
        return -q


def test_smc_evaluation_counts():
    particles = 2
    iterations = 2
    mutations = 2
    leapfrog_steps = 3

    config = {
        "seed": 42,
        "sample_shape": (particles, 2),
        "initial_sampler_cl": lambda key, shape: jax.random.normal(key, shape=shape + (2,)),
        "maxiter": iterations,
        "numsteps": mutations,
        "numleapfrog": leapfrog_steps,
        "step_size": 0.01,
        "log_freq": iterations + 1,
        "density_cl": GaussianTarget(),
        "ESS_thres": 0.0,
        "tempering_fn": lambda t: t / iterations,
    }

    sampler = SMC(particles, iterations, config)

    assert sampler.evaluation_counts == {"density": 0, "gradient": 0}

    sampler.sample()

    assert sampler.evaluation_counts == {
        "density": 2 * particles * iterations * (1 + mutations),
        "gradient": particles * iterations * mutations * (leapfrog_steps + 1),
    }

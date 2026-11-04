"""Regression tests for Adjoint Sampling target evaluations."""

import jax.numpy as jnp
import numpy as np

from types import SimpleNamespace

from rmc.modules.adjoint import AdjointSampler


class CountingTarget:
    def __init__(self):
        self.gradient_calls = 0

    def der_log_target_proposal(self, x, tempering=1.0):
        self.gradient_calls += x.shape[0]
        return -tempering * x


def test_terminal_gradient_target_calls():
    target = CountingTarget()

    sampler = SimpleNamespace(
        Dcl=target,
        base_terminal_variance=2.0,
    )

    x = jnp.ones((4, 2))
    gradient = AdjointSampler.eval_terminal_gradient(sampler, x)

    np.testing.assert_allclose(gradient, 0.5 * x)
    assert target.gradient_calls == 4

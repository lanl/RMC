"""Tests for SMC log-normalizing-constant estimation."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rmc.modules.sampler import SMC


def estimate(accumulated, log_weights):
    state = SimpleNamespace(
        logZ=accumulated,
        logw=jnp.asarray(log_weights),
    )
    return SMC.log_z_estimate.fget(state)


def test_initial_normalized_weights():
    n = 8
    log_weights = jnp.full(n, -jnp.log(float(n)))
    np.testing.assert_allclose(estimate(0.0, log_weights), 0.0, atol=1e-6)


def test_unresampled_weight_contribution():
    # Initial weights sum to one. A constant increment c must add c to log Z.
    n = 5
    c = 1.7
    log_weights = jnp.full(n, -jnp.log(float(n)) + c)
    np.testing.assert_allclose(estimate(0.0, log_weights), c, atol=1e-6)


def test_accumulated_resampling_contribution():
    # Earlier stages contribute a, current weights contribute b.
    n = 7
    a = 1.25
    b = -0.4
    log_weights = jnp.full(n, -jnp.log(float(n)) + b)

    np.testing.assert_allclose(
        estimate(a, log_weights),
        a + b,
        atol=1e-6,
    )


def test_nonuniform_weights():
    weights = jnp.array([0.2, 0.3, 0.5])
    increments = jnp.array([-0.3, 0.4, 1.1])

    expected = float(jax.scipy.special.logsumexp(jnp.log(weights) + increments))

    np.testing.assert_allclose(
        estimate(0.0, jnp.log(weights) + increments),
        expected,
        atol=1e-6,
    )


def test_estimator_does_not_mutate_weights():
    state = SimpleNamespace(
        logZ=1.0,
        logw=jnp.array([-2.0, -1.0, -3.0]),
    )
    original = state.logw.copy()

    _ = SMC.log_z_estimate.fget(state)

    np.testing.assert_array_equal(state.logw, original)
    assert state.logZ == 1.0

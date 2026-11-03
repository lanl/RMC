"""Tests for importance-weight effective sample size."""

import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.importance import importance_ess


def test_uniform_weights():
    np.testing.assert_allclose(
        importance_ess(jnp.ones(100)),
        100.0,
        rtol=1e-6,
    )


def test_single_nonzero_weight():
    np.testing.assert_allclose(
        importance_ess(jnp.array([0.0, 0.0, 5.0, 0.0])),
        1.0,
    )


def test_exact_nonuniform_weights():
    # (1+2+3)^2 / (1+4+9) = 36/14.
    np.testing.assert_allclose(
        importance_ess(jnp.array([1.0, 2.0, 3.0])),
        36.0 / 14.0,
        rtol=1e-6,
    )


def test_scale_invariance():
    weights = jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        importance_ess(weights),
        importance_ess(1000.0 * weights),
        rtol=1e-6,
    )


def test_log_weights_agree_with_weights():
    weights = jnp.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(
        importance_ess(weights),
        importance_ess(jnp.log(weights), log_weights=True),
        rtol=1e-6,
    )


def test_large_log_weights():
    log_weights = jnp.array([1000.0, 1001.0, 1002.0])

    result = importance_ess(log_weights, log_weights=True)
    reference = importance_ess(jnp.exp(log_weights - 1002.0))

    np.testing.assert_allclose(result, reference, rtol=1e-4)


def test_negative_infinite_log_weights():
    log_weights = jnp.array([-jnp.inf, 0.0, -jnp.inf])

    np.testing.assert_allclose(
        importance_ess(log_weights, log_weights=True),
        1.0,
    )


def test_permutation_invariance():
    weights = jnp.array([1.0, 4.0, 2.0, 3.0])

    np.testing.assert_allclose(
        importance_ess(weights),
        importance_ess(weights[::-1]),
    )


@pytest.mark.parametrize(
    "weights",
    [
        jnp.array([]),
        jnp.ones((2, 2)),
        jnp.array([0.0, 0.0]),
        jnp.array([-1.0, 2.0]),
        jnp.array([jnp.nan, 1.0]),
        jnp.array([jnp.inf, 1.0]),
    ],
)
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        importance_ess(weights)


@pytest.mark.parametrize(
    "weights",
    [
        jnp.array([-jnp.inf, -jnp.inf]),
        jnp.array([jnp.nan, 0.0]),
        jnp.array([jnp.inf, 0.0]),
    ],
)
def test_invalid_log_weights(weights):
    with pytest.raises(ValueError):
        importance_ess(weights, log_weights=True)

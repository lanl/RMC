"""Tests for empirical log-density diagnostics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.density import (
    log_density_difference,
    log_density_error,
    mean_log_density,
)


def gaussian_log_density(x):
    return -0.5 * jnp.sum(x**2)


def test_mean_log_density():
    x = jnp.array([[0.0, 0.0], [3.0, 4.0]])

    np.testing.assert_allclose(
        mean_log_density(x, gaussian_log_density),
        -6.25,
    )


def test_signed_difference():
    x = jnp.array([[0.0, 0.0]])
    y = jnp.array([[3.0, 4.0]])

    np.testing.assert_allclose(
        log_density_difference(x, y, gaussian_log_density),
        12.5,
    )
    np.testing.assert_allclose(
        log_density_difference(y, x, gaussian_log_density),
        -12.5,
    )


def test_absolute_difference():
    x = jnp.array([[0.0, 0.0]])
    y = jnp.array([[3.0, 4.0]])

    np.testing.assert_allclose(
        log_density_error(x, y, gaussian_log_density),
        12.5,
    )
    np.testing.assert_allclose(
        log_density_error(y, x, gaussian_log_density),
        12.5,
    )


def test_additive_constant_invariance():
    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[2.0], [3.0]])

    def shifted_log_density(z):
        return gaussian_log_density(z) + 100.0

    original = log_density_difference(x, y, gaussian_log_density)
    shifted = log_density_difference(x, y, shifted_log_density)

    np.testing.assert_allclose(original, shifted, rtol=1e-6, atol=1e-5)


def test_unequal_sample_counts():
    x = jnp.array([[0.0], [0.0]])
    y = jnp.array([[1.0], [1.0], [1.0]])

    np.testing.assert_allclose(
        log_density_error(x, y, gaussian_log_density),
        0.5,
    )


def test_identical_collections():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0]])

    np.testing.assert_allclose(
        log_density_difference(x, x[::-1], gaussian_log_density),
        0.0,
    )


def test_jit():
    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[2.0], [3.0]])

    compiled = jax.jit(lambda a, b: log_density_difference(a, b, gaussian_log_density))(x, y)

    direct = log_density_difference(x, y, gaussian_log_density)

    np.testing.assert_allclose(compiled, direct, rtol=1e-6)


def test_matches_generic_observable_error():
    from rmc.metrics.observables import observable_error

    x = jnp.array([[0.0], [1.0]])
    y = jnp.array([[2.0], [3.0]])

    direct = log_density_error(x, y, gaussian_log_density)
    generic = observable_error(x, y, gaussian_log_density)

    np.testing.assert_allclose(direct, generic, rtol=1e-6)


@pytest.mark.parametrize(
    "samples",
    [
        jnp.ones((3,)),
        jnp.ones((0, 2)),
        jnp.ones((3, 0)),
    ],
)
def test_invalid_samples(samples):
    with pytest.raises(ValueError):
        mean_log_density(samples, gaussian_log_density)


def test_dimension_mismatch():
    x = jnp.ones((3, 2))
    y = jnp.ones((4, 3))

    with pytest.raises(ValueError):
        log_density_difference(x, y, gaussian_log_density)


def test_invalid_log_density_shape():
    x = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="scalar"):
        mean_log_density(x, lambda z: z)

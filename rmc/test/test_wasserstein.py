"""Tests for exact empirical Wasserstein distances."""

import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.wasserstein import wasserstein


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_one_dimensional_exact(p):
    x = jnp.array([[0.0], [1.0], [3.0], [5.0]])
    y = jnp.array([[2.0], [2.0], [4.0], [8.0]])

    expected = np.mean(np.array([2.0, 1.0, 1.0, 3.0]) ** p) ** (1.0 / p)

    np.testing.assert_allclose(wasserstein(x, y, p=p), expected, rtol=1e-6)


def test_two_dimensional_exact_assignment():
    x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    y = jnp.array([[1.0, 1.0], [0.0, 1.0]])

    result = wasserstein(x, y)

    np.testing.assert_allclose(result, 1.0, rtol=1e-6)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_translation(p):
    x = jnp.array([[0.0, 0.0], [1.0, 2.0], [-2.0, 1.0]])
    shift = jnp.array([3.0, 4.0])
    y = x + shift

    result = wasserstein(x, y, p=p)

    np.testing.assert_allclose(result, 5.0, rtol=1e-6)


def test_identical_empirical_distributions():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = x[jnp.array([2, 0, 1])]

    np.testing.assert_allclose(wasserstein(x, y), 0.0, atol=1e-6)


def test_symmetry():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = jnp.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])

    np.testing.assert_allclose(
        wasserstein(x, y),
        wasserstein(y, x),
        atol=1e-6,
    )


def test_permutation_invariance():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    original = wasserstein(x, y)
    permuted = wasserstein(
        x[jnp.array([2, 0, 1])],
        y[jnp.array([1, 2, 0])],
    )

    np.testing.assert_allclose(original, permuted, atol=1e-6)


def test_single_sample():
    x = jnp.array([[0.0, 0.0]])
    y = jnp.array([[3.0, 4.0]])

    np.testing.assert_allclose(wasserstein(x, y), 5.0, rtol=1e-6)


@pytest.mark.parametrize(
    "x,y",
    [
        (jnp.ones((3, 2)), jnp.ones((4, 2))),
        (jnp.ones((3, 2)), jnp.ones((3, 3))),
        (jnp.ones((3,)), jnp.ones((3,))),
        (jnp.ones((0, 2)), jnp.ones((0, 2))),
        (jnp.ones((3, 0)), jnp.ones((3, 0))),
    ],
)
def test_invalid_shapes(x, y):
    with pytest.raises(ValueError):
        wasserstein(x, y)


@pytest.mark.parametrize("p", [0.0, 0.5, -1.0, np.inf, np.nan])
def test_invalid_order(p):
    x = jnp.ones((3, 2))

    with pytest.raises(ValueError):
        wasserstein(x, x, p=p)


def test_nonfinite_samples():
    x = jnp.array([[0.0, 0.0], [jnp.nan, 1.0]])
    y = jnp.ones((2, 2))

    with pytest.raises(ValueError):
        wasserstein(x, y)

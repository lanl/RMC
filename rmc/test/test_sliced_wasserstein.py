"""Tests for sliced Wasserstein distances."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from rmc.metrics.sliced_wasserstein import sliced_wasserstein


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0])
def test_one_dimensional_exact(p):
    x = jnp.array([[0.0], [1.0], [3.0], [5.0]])
    y = jnp.array([[2.0], [2.0], [4.0], [8.0]])
    key = jax.random.key(0)

    expected = np.mean(np.array([2.0, 1.0, 1.0, 3.0]) ** p) ** (1.0 / p)

    result = sliced_wasserstein(x, y, key, p=p, num_projections=32)

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_identical_empirical_distributions():
    x = jnp.array([[0.0, 1.0], [2.0, -1.0], [3.0, 4.0]])
    y = x[jnp.array([2, 0, 1])]
    key = jax.random.key(1)

    result = sliced_wasserstein(x, y, key)

    np.testing.assert_allclose(result, 0.0, atol=1e-6)


def test_symmetry():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = jnp.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    key = jax.random.key(2)

    forward = sliced_wasserstein(x, y, key)
    backward = sliced_wasserstein(y, x, key)

    np.testing.assert_allclose(forward, backward, atol=1e-6)


def test_permutation_invariance():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    key = jax.random.key(3)

    original = sliced_wasserstein(x, y, key)
    permuted = sliced_wasserstein(
        x[jnp.array([2, 0, 1])],
        y[jnp.array([1, 2, 0])],
        key,
    )

    np.testing.assert_allclose(original, permuted, atol=1e-6)


def test_reproducibility():
    key = jax.random.key(4)
    x = jax.random.normal(jax.random.key(5), (20, 3))
    y = jax.random.normal(jax.random.key(6), (20, 3))

    first = sliced_wasserstein(x, y, key)
    second = sliced_wasserstein(x, y, key)

    np.testing.assert_array_equal(first, second)


def test_jit():
    x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    y = jnp.array([[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]])
    key = jax.random.key(7)

    compiled = jax.jit(sliced_wasserstein)(x, y, key)
    direct = sliced_wasserstein(x, y, key)

    np.testing.assert_allclose(compiled, direct, atol=1e-6)


def test_zero_distance():
    x = jnp.zeros((5, 3))
    y = jnp.zeros((5, 3))

    result = sliced_wasserstein(x, y, jax.random.key(8))

    np.testing.assert_allclose(result, 0.0, atol=1e-7)


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
        sliced_wasserstein(x, y, jax.random.key(9))


def test_invalid_parameters():
    x = jnp.ones((3, 2))
    key = jax.random.key(10)

    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, key, p=0.5)

    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, key, num_projections=0)

    with pytest.raises(ValueError):
        sliced_wasserstein(x, x, key, num_projections=1.5)

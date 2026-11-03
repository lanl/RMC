"""Tests for maximum mean discrepancy metrics."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from rmc.metrics.mmd import mmd_squared


@pytest.mark.parametrize(
    "kernel,nu",
    [
        ("gaussian", 1.5),
        ("matern", 0.5),
        ("matern", 1.5),
        ("matern", 2.5),
    ],
)
def test_constant_samples(kernel, nu):
    x = jnp.zeros((4, 2))
    y = jnp.ones((5, 2))

    result = mmd_squared(x, y, kernel=kernel, nu=nu)

    r = np.sqrt(2.0)
    if kernel == "gaussian":
        cross = np.exp(-1.0)
    elif nu == 0.5:
        cross = np.exp(-r)
    elif nu == 1.5:
        a = np.sqrt(3.0) * r
        cross = (1.0 + a) * np.exp(-a)
    else:
        a = np.sqrt(5.0) * r
        cross = (1.0 + a + a**2 / 3.0) * np.exp(-a)

    np.testing.assert_allclose(result, 2.0 * (1.0 - cross), rtol=1e-5)


@pytest.mark.parametrize("kernel", ["gaussian", "matern"])
def test_symmetry(kernel):
    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[3.0], [4.0], [5.0], [6.0]])

    a = mmd_squared(x, y, kernel=kernel)
    b = mmd_squared(y, x, kernel=kernel)

    np.testing.assert_allclose(a, b, atol=1e-6)


def test_permutation_invariance():
    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[3.0], [4.0], [5.0], [6.0]])

    original = mmd_squared(x, y)
    permuted = mmd_squared(x[jnp.array([2, 0, 1])], y[jnp.array([3, 1, 0, 2])])

    np.testing.assert_allclose(original, permuted, atol=1e-6)


def test_negative_unbiased_estimate():
    x = jnp.array([[0.0], [2.0]])
    y = jnp.array([[0.0], [2.0]])

    result = mmd_squared(x, y)

    assert result < 0.0


def test_jit():
    x = jnp.array([[0.0], [1.0], [2.0]])
    y = jnp.array([[1.0], [2.0], [3.0]])

    compiled = jax.jit(mmd_squared)(x, y)
    direct = mmd_squared(x, y)

    np.testing.assert_allclose(compiled, direct, atol=1e-6)


@pytest.mark.parametrize(
    "x,y",
    [
        (jnp.ones((1, 2)), jnp.ones((3, 2))),
        (jnp.ones((3, 2)), jnp.ones((3, 3))),
        (jnp.ones((3,)), jnp.ones((3, 1))),
    ],
)
def test_invalid_shapes(x, y):
    with pytest.raises(ValueError):
        mmd_squared(x, y)


def test_invalid_parameters():
    x = jnp.ones((3, 2))

    with pytest.raises(ValueError):
        mmd_squared(x, x, length_scale=0.0)

    with pytest.raises(ValueError):
        mmd_squared(x, x, kernel="unknown")

    with pytest.raises(ValueError):
        mmd_squared(x, x, kernel="matern", nu=1.0)

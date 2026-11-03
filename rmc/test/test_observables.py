"""Tests for empirical observable and moment errors."""

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from rmc.metrics.observables import moment_errors, observable_error


def test_scalar_observable():
    x = jnp.array([[0.0], [2.0], [4.0]])
    y = jnp.array([[1.0], [3.0], [5.0]])

    result = observable_error(x, y, lambda z: z[0])

    np.testing.assert_allclose(result, 1.0, rtol=1e-6)


def test_vector_observable():
    x = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    y = jnp.array([[3.0, 4.0], [5.0, 4.0]])

    result = observable_error(x, y, lambda z: z)

    np.testing.assert_allclose(result, 5.0, rtol=1e-6)


def test_relative_observable_error():
    x = jnp.array([[1.0], [3.0]])
    y = jnp.array([[2.0], [6.0]])

    result = observable_error(x, y, lambda z: z[0], relative=True)

    np.testing.assert_allclose(result, 0.5, rtol=1e-6)


def test_matrix_observable():
    x = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    y = jnp.array([[0.0, 1.0], [0.0, 1.0]])

    result = observable_error(x, y, lambda z: jnp.outer(z, z))

    np.testing.assert_allclose(result, np.sqrt(2.0), rtol=1e-6)


def test_unequal_sample_counts():
    x = jnp.array([[0.0], [2.0]])
    y = jnp.array([[1.0], [2.0], [3.0], [4.0]])

    result = observable_error(x, y, lambda z: z[0])
    moments = moment_errors(x, y)

    np.testing.assert_allclose(result, 1.5, rtol=1e-6)
    np.testing.assert_allclose(moments["mean"], 1.5, rtol=1e-6)


def test_exact_moment_errors():
    x = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    y = jnp.array([[0.0, 0.0], [0.0, 2.0]])

    result = moment_errors(x, y)

    np.testing.assert_allclose(result["mean"], np.sqrt(2.0), rtol=1e-6)
    np.testing.assert_allclose(
        result["second_moment"],
        2.0 * np.sqrt(2.0),
        rtol=1e-6,
    )


def test_relative_moment_errors():
    x = jnp.array([[1.0], [1.0]])
    y = jnp.array([[2.0], [2.0]])

    result = moment_errors(x, y, relative=True)

    np.testing.assert_allclose(result["mean"], 0.5, rtol=1e-6)
    np.testing.assert_allclose(result["second_moment"], 0.75, rtol=1e-6)


def test_identical_distributions():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = x[jnp.array([1, 0])]

    result = moment_errors(x, y)
    observable = observable_error(x, y, lambda z: jnp.sum(z**2))

    np.testing.assert_allclose(result["mean"], 0.0, atol=1e-6)
    np.testing.assert_allclose(result["second_moment"], 0.0, atol=1e-6)
    np.testing.assert_allclose(observable, 0.0, atol=1e-6)


def test_zero_reference_relative_error():
    x = jnp.array([[1.0], [1.0]])
    y = jnp.array([[0.0], [0.0]])

    result = observable_error(x, y, lambda z: z[0], relative=True)
    zero = observable_error(y, y, lambda z: z[0], relative=True)

    assert bool(jnp.isinf(result))
    np.testing.assert_allclose(zero, 0.0)


def test_permutation_invariance():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    y = jnp.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])

    original = moment_errors(x, y)
    permuted = moment_errors(x[jnp.array([2, 0, 1])], y[jnp.array([1, 2, 0])])

    for name in original:
        np.testing.assert_allclose(original[name], permuted[name], atol=1e-6)


def test_jit():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[2.0, 1.0], [4.0, 3.0]])

    compiled = jax.jit(moment_errors)(x, y)
    direct = moment_errors(x, y)

    for name in direct:
        np.testing.assert_allclose(compiled[name], direct[name], atol=1e-6)

    observable = lambda a, b: observable_error(a, b, lambda z: jnp.sum(z**2))

    np.testing.assert_allclose(
        jax.jit(observable)(x, y),
        observable(x, y),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "x,y",
    [
        (jnp.ones((3,)), jnp.ones((3, 1))),
        (jnp.ones((3, 2)), jnp.ones((4, 3))),
        (jnp.ones((0, 2)), jnp.ones((3, 2))),
        (jnp.ones((3, 0)), jnp.ones((3, 0))),
    ],
)
def test_invalid_shapes(x, y):
    with pytest.raises(ValueError):
        moment_errors(x, y)

    with pytest.raises(ValueError):
        observable_error(x, y, lambda z: z[0])

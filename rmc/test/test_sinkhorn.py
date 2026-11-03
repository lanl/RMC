"""Tests for debiased Sinkhorn divergence."""

import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.sinkhorn import sinkhorn_divergence
from rmc.metrics.wasserstein import wasserstein


def test_point_masses():
    x = jnp.array([[0.0, 0.0]])
    y = jnp.array([[3.0, 4.0]])

    for epsilon in (0.1, 1.0, 10.0):
        result = sinkhorn_divergence(x, y, epsilon=epsilon)
        np.testing.assert_allclose(result, 25.0, rtol=1e-6)


@pytest.mark.parametrize("epsilon", [0.1, 0.5, 2.0])
def test_translation(epsilon):
    x = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    y = x + jnp.array([0.0, 1.0])

    result = sinkhorn_divergence(x, y, epsilon=epsilon)

    np.testing.assert_allclose(result, 1.0, rtol=1e-4, atol=1e-5)


def test_identical_distributions():
    x = jnp.array([[0.0, 0.0], [2.0, 1.0], [4.0, -1.0]])

    result = sinkhorn_divergence(x, x)

    np.testing.assert_allclose(result, 0.0, atol=1e-5)


def test_symmetry():
    x = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    y = jnp.array([[0.5, 1.0], [2.0, 2.0], [4.0, 0.0]])

    forward = sinkhorn_divergence(x, y, epsilon=2.0)
    backward = sinkhorn_divergence(y, x, epsilon=2.0)

    np.testing.assert_allclose(forward, backward, atol=1e-4)


def test_permutation_invariance():
    x = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    y = jnp.array([[0.5, 1.0], [2.0, 2.0], [4.0, 0.0]])

    original = sinkhorn_divergence(x, y, epsilon=2.0)
    permuted = sinkhorn_divergence(
        x[jnp.array([2, 0, 1])],
        y[jnp.array([1, 2, 0])],
        epsilon=2.0,
    )

    np.testing.assert_allclose(original, permuted, atol=1e-4)


def test_small_regularization_matches_exact_ot():
    x = jnp.array([[0.0, 0.0], [2.0, 0.0]])
    y = jnp.array([[0.0, 1.0], [3.0, 1.0]])

    exact_squared = float(wasserstein(x, y) ** 2)
    result = float(sinkhorn_divergence(x, y, epsilon=0.05))

    np.testing.assert_allclose(result, exact_squared, atol=0.05)


def test_integer_samples():
    x = jnp.array([[0, 0], [2, 0]])
    y = jnp.array([[0, 1], [2, 1]])

    np.testing.assert_allclose(
        sinkhorn_divergence(x, y),
        1.0,
        rtol=1e-4,
    )


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
        sinkhorn_divergence(x, y)


def test_invalid_parameters():
    x = jnp.ones((3, 2))

    for epsilon in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            sinkhorn_divergence(x, x, epsilon=epsilon)

    with pytest.raises(ValueError):
        sinkhorn_divergence(x, x, tolerance=0.0)

    with pytest.raises(ValueError):
        sinkhorn_divergence(x, x, max_iterations=0)


def test_nonfinite_samples():
    x = jnp.array([[0.0, 0.0], [jnp.nan, 1.0]])
    y = jnp.ones((2, 2))

    with pytest.raises(ValueError):
        sinkhorn_divergence(x, y)


def test_nonconvergence():
    x = jnp.array([[0.0], [1.0], [3.0]])
    y = jnp.array([[0.0], [2.0], [5.0]])

    with pytest.raises(RuntimeError, match="did not converge"):
        sinkhorn_divergence(
            x,
            y,
            epsilon=0.5,
            tolerance=1e-8,
            max_iterations=1,
        )


def test_small_regularization_convergence_limit():
    """A concentrated coupling must not silently pass convergence checks."""
    x = jnp.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    y = jnp.array([[0.5, 1.0], [2.0, 2.0], [4.0, 0.0]])

    with pytest.raises(RuntimeError, match="did not converge"):
        sinkhorn_divergence(
            x,
            y,
            epsilon=0.5,
            tolerance=1e-5,
            max_iterations=1000,
        )

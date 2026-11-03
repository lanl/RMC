"""Tests for Fisher score discrepancy."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.scores import fisher_score_discrepancy


def test_identical_scores():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0]])

    result = fisher_score_discrepancy(
        x,
        lambda z: -z,
        lambda z: -z,
    )

    np.testing.assert_allclose(result, 0.0)


def test_gaussian_mean_shift():
    x = jnp.array([[0.0, 0.0], [1.0, -1.0], [2.0, 3.0]])
    shift = jnp.array([3.0, 4.0])

    result = fisher_score_discrepancy(
        x,
        target_score=lambda z: -z,
        approximate_score=lambda z: -(z - shift),
    )

    np.testing.assert_allclose(result, 12.5, rtol=1e-6)


def test_one_dimensional_example():
    x = jnp.array([[-2.0], [0.0], [2.0]])

    result = fisher_score_discrepancy(
        x,
        target_score=lambda z: -z,
        approximate_score=lambda z: -2.0 * z,
    )

    # Mean(x**2) = 8/3; multiply by 1/2.
    np.testing.assert_allclose(result, 4.0 / 3.0, rtol=1e-6)


def test_permutation_invariance():
    x = jnp.array([[1.0], [2.0], [3.0], [4.0]])

    def compute(samples):
        return fisher_score_discrepancy(
            samples,
            lambda z: -z,
            lambda z: -2.0 * z,
        )

    np.testing.assert_allclose(compute(x), compute(x[::-1]))


def test_jit():
    x = jnp.array([[0.0, 1.0], [2.0, 3.0]])

    def evaluate(samples):
        return fisher_score_discrepancy(
            samples,
            target_score=lambda z: -z,
            approximate_score=lambda z: -2.0 * z,
        )

    np.testing.assert_allclose(
        jax.jit(evaluate)(x),
        evaluate(x),
        rtol=1e-6,
    )


def test_score_from_log_density():
    x = jnp.array([[0.0, 0.0], [1.0, 2.0]])
    shift = jnp.array([1.0, -2.0])

    def target_log_density(z):
        return -0.5 * jnp.sum(z**2)

    def approximate_log_density(z):
        return -0.5 * jnp.sum((z - shift) ** 2)

    result = fisher_score_discrepancy(
        x,
        jax.grad(target_log_density),
        jax.grad(approximate_log_density),
    )

    np.testing.assert_allclose(result, 2.5, rtol=1e-6)


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
        fisher_score_discrepancy(samples, lambda z: -z, lambda z: -z)


def test_invalid_target_score_shape():
    x = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="target_score"):
        fisher_score_discrepancy(
            x,
            target_score=lambda z: jnp.sum(z),
            approximate_score=lambda z: -z,
        )


def test_invalid_approximate_score_shape():
    x = jnp.ones((3, 2))

    with pytest.raises(ValueError, match="approximate_score"):
        fisher_score_discrepancy(
            x,
            target_score=lambda z: -z,
            approximate_score=lambda z: jnp.sum(z),
        )

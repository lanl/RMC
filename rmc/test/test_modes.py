"""Tests for ball-based discrete mode metrics."""

import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.modes import (
    discrete_tv,
    intermode_mass,
    mode_coverage,
    mode_region_probabilities,
)


@pytest.fixture
def centers():
    coordinates = jnp.array([-4.0, 0.0, 4.0])
    gx, gy = jnp.meshgrid(coordinates, coordinates)
    return jnp.stack((gx.ravel(), gy.ravel()), axis=1)


def test_ten_regions(centers):
    samples = jnp.concatenate(
        (centers, jnp.array([[20.0, 20.0]])),
        axis=0,
    )

    probabilities = mode_region_probabilities(samples, centers, radius=1.0)

    assert probabilities.shape == (10,)
    np.testing.assert_allclose(probabilities, jnp.ones(10) / 10)
    np.testing.assert_allclose(jnp.sum(probabilities), 1.0, atol=1e-6)


def test_full_coverage(centers):
    np.testing.assert_allclose(mode_coverage(centers, centers, 1.0), 1.0)
    np.testing.assert_allclose(intermode_mass(centers, centers, 1.0), 0.0)


def test_partial_coverage(centers):
    samples = centers[:6]

    np.testing.assert_allclose(
        mode_coverage(samples, centers, 1.0),
        6.0 / 9.0,
    )


def test_interregion_samples_do_not_count_as_coverage(centers):
    samples = jnp.array([[20.0, 20.0], [30.0, 30.0]])

    np.testing.assert_allclose(mode_coverage(samples, centers, 1.0), 0.0)
    np.testing.assert_allclose(intermode_mass(samples, centers, 1.0), 1.0)


def test_tenth_region_contributes_to_tv(centers):
    reference = centers
    generated = jnp.concatenate(
        (centers[:8], jnp.array([[20.0, 20.0]])),
        axis=0,
    )

    # One mode loses mass 1/9, and the complementary region gains 1/9.
    np.testing.assert_allclose(
        discrete_tv(generated, reference, centers, 1.0),
        1.0 / 9.0,
    )


def test_missing_three_modes(centers):
    generated = centers[:6]
    reference = centers

    np.testing.assert_allclose(
        discrete_tv(generated, reference, centers, 1.0),
        1.0 / 3.0,
    )


def test_identical_empirical_distributions(centers):
    samples = jnp.concatenate(
        (centers, jnp.array([[20.0, 20.0]])),
        axis=0,
    )

    np.testing.assert_allclose(
        discrete_tv(samples, samples[::-1], centers, 1.0),
        0.0,
    )


def test_symmetry(centers):
    x = centers[:6]
    y = centers[3:]

    np.testing.assert_allclose(
        discrete_tv(x, y, centers, 1.0),
        discrete_tv(y, x, centers, 1.0),
    )


def test_boundary(centers):
    samples = jnp.array([[-3.0, -4.0], [-2.9, -4.0]])

    probabilities = mode_region_probabilities(samples, centers, 1.0)

    np.testing.assert_allclose(probabilities[0], 0.5)
    np.testing.assert_allclose(probabilities[-1], 0.5)


def test_unequal_sample_counts(centers):
    x = centers
    y = jnp.concatenate((centers, centers), axis=0)

    np.testing.assert_allclose(discrete_tv(x, y, centers, 1.0), 0.0)


def test_overlapping_balls():
    centers = jnp.array([[0.0], [1.0]])
    samples = jnp.array([[0.0]])

    with pytest.raises(ValueError, match="disjoint"):
        mode_region_probabilities(samples, centers, 0.6)


def test_invalid_radius(centers):
    with pytest.raises(ValueError):
        mode_coverage(centers, centers, 0.0)


def test_invalid_dimensions(centers):
    with pytest.raises(ValueError):
        mode_coverage(jnp.ones((3, 3)), centers, 1.0)


def test_nonfinite_samples(centers):
    with pytest.raises(ValueError):
        mode_coverage(jnp.array([[jnp.nan, 0.0]]), centers, 1.0)


def test_single_mode_and_complement():
    centers = jnp.array([[0.0, 0.0]])
    samples = jnp.array([[0.0, 0.0], [3.0, 0.0]])

    probabilities = mode_region_probabilities(samples, centers, 1.0)

    np.testing.assert_allclose(probabilities, [0.5, 0.5])
    np.testing.assert_allclose(mode_coverage(samples, centers, 1.0), 1.0)
    np.testing.assert_allclose(intermode_mass(samples, centers, 1.0), 0.5)

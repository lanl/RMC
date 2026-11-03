"""Tests for autocorrelation-based MCMC effective sample size."""

import jax.numpy as jnp
import numpy as np
import pytest

from rmc.metrics.mcmc import mcmc_ess


def test_independent_gaussian_chain():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=4000)

    ess = float(mcmc_ess(samples))

    assert 1500.0 <= ess <= 4000.0


def test_autocorrelated_ar1_chain():
    rng = np.random.default_rng(42)
    n = 6000
    phi = 0.8

    innovations = rng.normal(size=n)
    samples = np.empty(n)
    samples[0] = innovations[0]

    for i in range(1, n):
        samples[i] = phi * samples[i - 1] + np.sqrt(1.0 - phi**2) * innovations[i]

    estimated = float(mcmc_ess(samples))
    theoretical = n * (1.0 - phi) / (1.0 + phi)

    # A broad statistical tolerance rather than an exact equality.
    assert 0.5 * theoretical < estimated < 1.8 * theoretical


def test_multidimensional_input_rejected():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(1000, 3))

    with pytest.raises(ValueError, match="one-dimensional"):
        mcmc_ess(samples)


def test_scalar_observable_of_vector_chain():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(1000, 3))

    # Evaluate f(x) = ||x||^2 before estimating ESS.
    observations = np.sum(samples**2, axis=1)

    result = mcmc_ess(observations)

    assert result.shape == ()
    assert bool(jnp.isfinite(result))


def test_scalar_output_shape():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=1000)

    assert mcmc_ess(samples).shape == ()


def test_constant_chain_returns_nan():
    samples = np.ones(100)

    assert bool(jnp.isnan(mcmc_ess(samples)))


def test_location_scale_invariance():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=1000)

    original = mcmc_ess(samples)
    transformed = mcmc_ess(3.0 * samples + 10.0)

    np.testing.assert_allclose(original, transformed, rtol=1e-5)


def test_ess_bounds():
    rng = np.random.default_rng(42)
    samples = rng.normal(size=1000)

    ess = float(mcmc_ess(samples))

    assert 1.0 <= ess <= 1000.0


@pytest.mark.parametrize(
    "samples",
    [
        np.array([]),
        np.array([1.0, 2.0, 3.0]),
        np.ones((5, 0)),
        np.ones((2, 2, 2)),
        np.array([1.0, np.nan, 2.0, 3.0]),
        np.array([1.0, np.inf, 2.0, 3.0]),
    ],
)
def test_invalid_chain(samples):
    with pytest.raises(ValueError):
        mcmc_ess(samples)

# -*- coding: utf-8 -*-

"""Autocorrelation-based effective sample size for a single MCMC chain."""

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _autocorrelation(x: np.ndarray) -> np.ndarray:
    """Estimate autocorrelation using biased FFT autocovariances."""
    n = x.size
    centered = x - np.mean(x)

    if not np.any(centered):
        return np.full(n, np.nan)

    fft_size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        transform * np.conjugate(transform),
        n=fft_size,
    )[:n]

    # Biased autocovariance normalization, common to all lags.
    return autocovariance / autocovariance[0]


def _single_chain_ess(x: np.ndarray) -> float:
    """Estimate ESS with Geyer's initial-positive/monotone sequence."""
    n = x.size
    rho = _autocorrelation(x)

    if np.isnan(rho[0]):
        return float("nan")

    # Gamma_j = rho_(2j) + rho_(2j+1).
    pairs = rho[0::2][: n // 2] + rho[1::2][: n // 2]

    # Retain only the initial strictly positive sequence.
    nonpositive = np.flatnonzero(pairs <= 0)
    if nonpositive.size:
        pairs = pairs[: nonpositive[0]]

    if pairs.size == 0:
        return float(n)

    # Enforce nonincreasing paired autocorrelation sums.
    pairs = np.minimum.accumulate(pairs)

    tau = 2.0 * np.sum(pairs) - 1.0

    # Conservative reporting convention: 1 <= ESS <= N.
    tau = np.clip(tau, 1.0, float(n))
    return float(n / tau)


def mcmc_ess(observations: ArrayLike) -> jnp.ndarray:
    """Estimate effective sample size of a scalar observable time series.

    Uses FFT autocorrelations and Geyer's initial-positive and
    initial-monotone sequence procedure.

    Args:
        observations: Ordered scalar observations with shape (N,),
            with N >= 4. For a vector-valued Markov chain, evaluate
            a scalar observable before calling this function.

    Returns:
        Scalar JAX array containing the estimated ESS.
        A constant time series returns NaN.

    Raises:
        ValueError: If observations are not one-dimensional, contain
            fewer than four entries, or contain nonfinite values.

    Notes:
        Assumes stationarity. Independent chains must not be concatenated.
        The estimate is capped to [1, N]. Computation uses NumPy FFTs
        on the CPU and is not JAX-JIT-compatible.
    """
    values = np.asarray(observations, dtype=np.float64)

    if values.ndim != 1 or values.shape[0] < 4:
        raise ValueError("observations must be a one-dimensional array with N >= 4.")

    if not np.all(np.isfinite(values)):
        raise ValueError("observations must contain only finite values.")

    return jnp.asarray(_single_chain_ess(values))

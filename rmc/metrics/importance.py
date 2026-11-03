# -*- coding: utf-8 -*-

"""Effective sample size for importance weights."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike


def importance_ess(
    weights: ArrayLike,
    log_weights: bool = False,
) -> jax.Array:
    """Compute effective sample size from importance weights.

    ESS = (sum_i w_i)^2 / sum_i w_i^2.

    Args:
        weights: One-dimensional nonnegative weights, or log weights
            when log_weights=True.
        log_weights: Interpret the input as logarithmic weights.

    Returns:
        Scalar JAX array containing ESS between 1 and N.

    Raises:
        ValueError: If weights are invalid or have zero total mass.

    Notes:
        Log weights may contain -inf, representing zero weights.
        Weights need not be normalized.
        This is weight-based ESS, not autocorrelation-based MCMC ESS.
        Input validation uses Python control flow, so this public
        function is not directly JIT-compatible.
    """
    values = jnp.asarray(weights)

    if values.ndim != 1 or values.shape[0] < 1:
        raise ValueError("weights must be a nonempty one-dimensional array.")

    values = values.astype(jnp.result_type(values, jnp.float32))

    if log_weights:
        if bool(jnp.any(jnp.isnan(values) | jnp.isposinf(values))):
            raise ValueError("Log weights must not contain NaN or +inf.")
        if not bool(jnp.any(jnp.isfinite(values))):
            raise ValueError("At least one log weight must be finite.")

        log_ess = 2.0 * logsumexp(values) - logsumexp(2.0 * values)
        ess = jnp.exp(log_ess)
    else:
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Weights must be finite.")
        if bool(jnp.any(values < 0)):
            raise ValueError("Weights must be nonnegative.")
        if not bool(jnp.any(values > 0)):
            raise ValueError("At least one weight must be positive.")

        # Normalize by the maximum to avoid overflow from large weights.
        scaled = values / jnp.max(values)
        ess = jnp.sum(scaled) ** 2 / jnp.sum(scaled**2)

    return jnp.clip(ess, 1.0, values.shape[0])

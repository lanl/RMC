# -*- coding: utf-8 -*-

"""Score-based discrepancies between probability distributions."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def fisher_score_discrepancy(
    samples: ArrayLike,
    target_score: Callable,
    approximate_score: Callable,
) -> jax.Array:
    """Estimate Fisher divergence using samples from the approximation.

    Computes 0.5 * E_q[||score_q(X) - score_target(X)||^2]
    using the empirical average over the supplied samples.

    Both score functions must accept one sample of shape (d,) and
    return a score vector of shape (d,).

    Args:
        samples: Samples from the approximate distribution, shape (N, d).
        target_score: Target score function, evaluated pointwise.
        approximate_score: Approximate distribution's score function.

    Returns:
        Scalar Monte Carlo estimate of Fisher divergence.

    Raises:
        ValueError: If samples or score outputs have invalid shapes.

    Notes:
        The score functions must be compatible with jax.vmap.
        Nonfinite score evaluations propagate to the result.
    """
    x = jnp.asarray(samples)

    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Samples must have nonempty shape (N, d).")

    x = x.astype(jnp.result_type(x, jnp.float32))

    target = jax.vmap(target_score)(x)
    approximate = jax.vmap(approximate_score)(x)

    if target.shape != x.shape:
        raise ValueError("target_score must return a vector of shape (d,).")
    if approximate.shape != x.shape:
        raise ValueError("approximate_score must return a vector of shape (d,).")

    squared_errors = jnp.sum((approximate - target) ** 2, axis=1)

    return 0.5 * jnp.mean(squared_errors)

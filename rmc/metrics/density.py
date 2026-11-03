# -*- coding: utf-8 -*-

"""Log-density diagnostics for generated and reference samples."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def mean_log_density(
    samples: ArrayLike,
    log_density: Callable,
) -> jax.Array:
    """Compute empirical mean of a scalar log-density function.

    Args:
        samples: Sample collection of shape (N, d).
        log_density: Function mapping a single sample of shape (d,)
            to a scalar log-density value.

    Returns:
        Scalar empirical mean log density.

    Raises:
        ValueError: If samples or log-density outputs have invalid shapes.

    Notes:
        The supplied function may be an unnormalized log density.
        Its additive normalization constant affects the returned mean.
        Nonfinite evaluations propagate to the returned result.
    """
    x = jnp.asarray(samples)

    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Samples must have nonempty shape (N, d).")

    x = x.astype(jnp.result_type(x, jnp.float32))
    values = jax.vmap(log_density)(x)

    if values.shape != (x.shape[0],):
        raise ValueError("log_density must return a scalar per sample.")

    return jnp.mean(values)


def log_density_difference(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    log_density: Callable,
) -> jax.Array:
    """Compute signed difference of empirical mean log densities.

    Returns the generated-sample mean minus the reference-sample mean.
    An additive constant in log_density cancels between the two means.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (M, d).
        log_density: Target log density evaluated on one sample.

    Returns:
        Scalar signed difference.
    """
    x = jnp.asarray(samples)
    y = jnp.asarray(reference_samples)

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Sample collections must have matching dimensions.")

    return mean_log_density(x, log_density) - mean_log_density(y, log_density)


def log_density_error(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    log_density: Callable,
) -> jax.Array:
    """Compute absolute difference of empirical mean log densities.

    The difference is invariant to additive log-normalization constants.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (M, d).
        log_density: Target log density evaluated on one sample.

    Returns:
        Scalar absolute discrepancy.
    """
    return jnp.abs(log_density_difference(samples, reference_samples, log_density))

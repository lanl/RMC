# -*- coding: utf-8 -*-

"""Errors of empirical observables and moments."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def _validate_samples(samples: ArrayLike, reference_samples: ArrayLike):
    """Validate sample arrays and use a common floating-point dtype."""
    x = jnp.asarray(samples)
    y = jnp.asarray(reference_samples)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sample collections must have shape (N, d).")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sample collections must have matching dimensions.")
    if min(x.shape[0], y.shape[0], x.shape[1]) < 1:
        raise ValueError("Sample collections must be nonempty.")

    dtype = jnp.result_type(x, y, jnp.float32)
    return x.astype(dtype), y.astype(dtype)


def _error(estimate: jax.Array, reference: jax.Array, relative: bool):
    """Compute absolute or relative Euclidean/Frobenius error."""
    absolute = jnp.linalg.norm(jnp.ravel(estimate - reference))

    if not relative:
        return absolute

    scale = jnp.linalg.norm(jnp.ravel(reference))
    return jnp.where(
        scale > 0,
        absolute / jnp.where(scale > 0, scale, 1.0),
        jnp.where(absolute == 0, 0.0, jnp.inf),
    )


def observable_error(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    observable: Callable,
    relative: bool = False,
) -> jax.Array:
    """Compute error in an empirical expectation of an observable.

    The observable is evaluated independently at each sample point.
    It may return a scalar, vector, or higher-dimensional array.
    Array-valued discrepancies use the Euclidean/Frobenius norm.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (M, d).
        observable: Function mapping one sample of shape (d,) to an array.
        relative: Whether to normalize by the reference expectation norm.

    Returns:
        Scalar absolute or relative error in the empirical expectation.
        A zero reference with nonzero discrepancy produces infinity.

    Raises:
        ValueError: If the sample collections have invalid shapes.
    """
    x, y = _validate_samples(samples, reference_samples)

    estimate = jnp.mean(jax.vmap(observable)(x), axis=0)
    reference = jnp.mean(jax.vmap(observable)(y), axis=0)

    return _error(estimate, reference, relative)


def moment_errors(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    relative: bool = False,
) -> dict:
    """Compute errors in first and raw second moments.

    The first moment is E[X], and the raw second moment is E[XX^T].
    Their errors use Euclidean and Frobenius norms, respectively.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (M, d).
        relative: Whether to normalize by the corresponding reference norm.

    Returns:
        Dictionary containing scalar JAX arrays under keys
        "mean" and "second_moment".

    Raises:
        ValueError: If the sample collections have invalid shapes.
    """
    x, y = _validate_samples(samples, reference_samples)

    mean_x = jnp.mean(x, axis=0)
    mean_y = jnp.mean(y, axis=0)

    second_x = (x.T @ x) / x.shape[0]
    second_y = (y.T @ y) / y.shape[0]

    return {
        "mean": _error(mean_x, mean_y, relative),
        "second_moment": _error(second_x, second_y, relative),
    }

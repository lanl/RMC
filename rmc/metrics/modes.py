# -*- coding: utf-8 -*-

"""Discrete discrepancies for disjoint balls around target modes."""

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _validate(samples, centers, radius):
    """Validate samples, centers, and disjoint mode balls."""
    x = jnp.asarray(samples)
    c = jnp.asarray(centers)

    if x.ndim != 2 or c.ndim != 2:
        raise ValueError("Samples and centers must be two-dimensional arrays.")
    if x.shape[0] < 1 or c.shape[0] < 1:
        raise ValueError("Samples and centers must be nonempty.")
    if x.shape[1] < 1 or x.shape[1] != c.shape[1]:
        raise ValueError("Samples and centers must have matching dimensions.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive.")

    if not bool(jnp.all(jnp.isfinite(x))) or not bool(jnp.all(jnp.isfinite(c))):
        raise ValueError("Samples and centers must be finite.")

    dtype = jnp.result_type(x, c, jnp.float32)
    x = x.astype(dtype)
    c = c.astype(dtype)

    distances = np.array(jnp.sqrt(jnp.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)))
    np.fill_diagonal(distances, np.inf)

    if np.any(distances <= 2.0 * radius):
        raise ValueError("Mode balls must be disjoint.")

    return x, c


def mode_region_probabilities(
    samples: ArrayLike,
    centers: ArrayLike,
    radius: float,
) -> jnp.ndarray:
    """Compute empirical probabilities of mode balls and their complement.

    The first K entries correspond to balls around the K centers.
    The final entry corresponds to all points outside those balls.

    Args:
        samples: Samples of shape (N, d).
        centers: Mode centers of shape (K, d).
        radius: Positive radius of each disjoint mode ball.

    Returns:
        Empirical probability vector of shape (K + 1,).
    """
    x, c = _validate(samples, centers, radius)

    squared_distances = jnp.sum(
        (x[:, None, :] - c[None, :, :]) ** 2,
        axis=-1,
    )

    nearest = jnp.argmin(squared_distances, axis=1)
    minimum_distance = jnp.min(squared_distances, axis=1)

    labels = jnp.where(
        minimum_distance <= radius**2,
        nearest,
        c.shape[0],
    )

    counts = jnp.bincount(labels, length=c.shape[0] + 1)
    return counts / x.shape[0]


def discrete_tv(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    centers: ArrayLike,
    radius: float,
) -> jnp.ndarray:
    """Compute discrete TV over mode balls and the inter-mode region.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (M, d).
        centers: Mode centers of shape (K, d).
        radius: Positive radius of each disjoint mode ball.

    Returns:
        Total variation between empirical (K + 1)-region distributions.
    """
    p = mode_region_probabilities(samples, centers, radius)
    q = mode_region_probabilities(reference_samples, centers, radius)

    return 0.5 * jnp.sum(jnp.abs(p - q))


def mode_coverage(
    samples: ArrayLike,
    centers: ArrayLike,
    radius: float,
) -> jnp.ndarray:
    """Compute fraction of prescribed mode balls containing samples.

    The complementary inter-mode region is excluded from coverage.
    """
    probabilities = mode_region_probabilities(samples, centers, radius)

    return jnp.mean((probabilities[:-1] > 0).astype(probabilities.dtype))


def intermode_mass(
    samples: ArrayLike,
    centers: ArrayLike,
    radius: float,
) -> jnp.ndarray:
    """Compute empirical mass outside all prescribed mode balls."""
    probabilities = mode_region_probabilities(samples, centers, radius)

    return probabilities[-1]

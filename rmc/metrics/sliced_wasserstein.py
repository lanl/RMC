# -*- coding: utf-8 -*-

"""Sliced Wasserstein distances between empirical sample distributions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def sliced_wasserstein(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    key: jax.Array,
    p: float = 2.0,
    num_projections: int = 128,
) -> jnp.ndarray:
    """Compute the Monte Carlo sliced Wasserstein distance.

    Projection directions are sampled independently and uniformly from
    the unit sphere. Equal sample counts allow exact one-dimensional
    empirical Wasserstein distances via sorting.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (N, d).
        key: JAX random key for sampling projection directions.
        p: Wasserstein order, at least 1.
        num_projections: Positive number of random projections.

    Returns:
        Scalar Monte Carlo estimate of the sliced Wasserstein distance.

    Raises:
        ValueError: If sample shapes or metric parameters are invalid.
    """
    x = jnp.asarray(samples)
    y = jnp.asarray(reference_samples)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sample collections must have shape (N, d).")
    if x.shape != y.shape:
        raise ValueError("Sample collections must have matching shapes.")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Sample collections must be nonempty.")
    if p < 1:
        raise ValueError("p must be at least 1.")
    if not isinstance(num_projections, int) or num_projections < 1:
        raise ValueError("num_projections must be a positive integer.")

    d = x.shape[1]

    directions = jax.random.normal(key, (num_projections, d))
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    projected_x = jnp.sort(x @ directions.T, axis=0)
    projected_y = jnp.sort(y @ directions.T, axis=0)

    return jnp.mean(jnp.abs(projected_x - projected_y) ** p) ** (1.0 / p)

# -*- coding: utf-8 -*-

"""Exact empirical Wasserstein distances between sample collections."""

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from rmc.utils.kernels import cdistSQ


def wasserstein(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    p: float = 2.0,
) -> jnp.ndarray:
    """Compute the exact empirical Wasserstein distance.

    Both sample collections have equal size and uniform empirical weights.
    The optimal transport problem is solved as a linear assignment problem
    using SciPy. The assignment runs on the CPU and is not JAX-JIT-compatible.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (N, d).
        p: Wasserstein order, at least 1.

    Returns:
        Scalar empirical Wasserstein distance W_p.

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
    if not np.isfinite(p) or p < 1:
        raise ValueError("p must be finite and at least 1.")

    if not bool(jnp.all(jnp.isfinite(x))) or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("Sample collections must contain finite values.")

    if x.shape[1] == 1:
        sorted_x = jnp.sort(x[:, 0])
        sorted_y = jnp.sort(y[:, 0])
        return jnp.mean(jnp.abs(sorted_x - sorted_y) ** p) ** (1.0 / p)

    squared_distances = np.asarray(cdistSQ(x, y))
    costs = np.maximum(squared_distances, 0.0) ** (p / 2.0)

    row_ind, col_ind = linear_sum_assignment(costs)

    return jnp.asarray(np.mean(costs[row_ind, col_ind]) ** (1.0 / p))

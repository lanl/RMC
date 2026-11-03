# -*- coding: utf-8 -*-

"""Entropic optimal transport and debiased Sinkhorn divergence."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from rmc.utils.kernels import cdistSQ


def _sinkhorn_cost(
    cost: jax.Array,
    epsilon: float,
    tolerance: float,
    max_iterations: int,
) -> jax.Array:
    """Compute regularized OT with KL relative to product marginals."""
    n, m = cost.shape
    log_a = -jnp.log(jnp.asarray(n, dtype=cost.dtype))
    log_b = -jnp.log(jnp.asarray(m, dtype=cost.dtype))
    log_k = -cost / epsilon

    def condition(state):
        iteration, _, _, residual = state
        return (iteration < max_iterations) & (residual > tolerance)

    def update(state):
        iteration, _, log_v, _ = state

        log_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_k + log_u[:, None], axis=0)

        log_p = log_u[:, None] + log_k + log_v[None, :]

        row_mass = jnp.exp(logsumexp(log_p, axis=1))
        col_mass = jnp.exp(logsumexp(log_p, axis=0))

        residual = jnp.maximum(
            jnp.max(jnp.abs(row_mass - jnp.exp(log_a))),
            jnp.max(jnp.abs(col_mass - jnp.exp(log_b))),
        )

        return iteration + 1, log_u, log_v, residual

    initial = (
        jnp.asarray(0),
        jnp.zeros((n,), dtype=cost.dtype),
        jnp.zeros((m,), dtype=cost.dtype),
        jnp.asarray(jnp.inf, dtype=cost.dtype),
    )

    iteration, log_u, log_v, residual = jax.lax.while_loop(condition, update, initial)

    if float(residual) > tolerance:
        raise RuntimeError(
            f"Sinkhorn did not converge within {int(iteration)} iterations; "
            f"marginal residual = {float(residual):.3e}."
        )

    log_p = log_u[:, None] + log_k + log_v[None, :]
    coupling = jnp.exp(log_p)

    transport = jnp.sum(coupling * cost)
    kl = jnp.sum(coupling * (log_p - log_a - log_b))

    return transport + epsilon * kl


def sinkhorn_divergence(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    epsilon: float = 0.1,
    tolerance: float = 1e-5,
    max_iterations: int = 1000,
) -> jax.Array:
    """Compute debiased Sinkhorn divergence using squared Euclidean cost.

    The empirical measures have uniform weights and equal sample counts.
    Entropic OT uses KL regularization relative to the product marginals.
    Sinkhorn updates are performed in the log domain.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Reference samples of shape (N, d).
        epsilon: Positive entropic regularization parameter.
        tolerance: Positive absolute marginal convergence tolerance.
        max_iterations: Maximum number of Sinkhorn iterations per OT problem.

    Returns:
        Scalar Sinkhorn divergence in squared-distance units.

    Raises:
        ValueError: If inputs or parameters are invalid.
        RuntimeError: If a Sinkhorn solve fails to converge.

    Notes:
        The final convergence check converts a JAX scalar to a Python
        float, so this function is not directly JIT-compatible.
    """
    x = jnp.asarray(samples)
    y = jnp.asarray(reference_samples)
    dtype = jnp.result_type(x, y, jnp.float32)
    x = x.astype(dtype)
    y = y.astype(dtype)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sample collections must have shape (N, d).")
    if x.shape != y.shape:
        raise ValueError("Sample collections must have matching shapes.")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("Sample collections must be nonempty.")

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive.")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")

    if not bool(jnp.all(jnp.isfinite(x))) or not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("Sample collections must contain finite values.")

    cxy = cdistSQ(x, y)
    cxx = cdistSQ(x, x)
    cyy = cdistSQ(y, y)

    cross = _sinkhorn_cost(cxy, epsilon, tolerance, max_iterations)
    self_x = _sinkhorn_cost(cxx, epsilon, tolerance, max_iterations)
    self_y = _sinkhorn_cost(cyy, epsilon, tolerance, max_iterations)

    return cross - 0.5 * (self_x + self_y)

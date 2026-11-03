# -*- coding: utf-8 -*-

"""Maximum mean discrepancy metrics for sample collections."""

import jax.numpy as jnp
from jax.typing import ArrayLike

from rmc.utils.kernels import cdistSQ


def _kernel_matrix(
    x: ArrayLike,
    y: ArrayLike,
    kernel: str,
    length_scale: float,
    nu: float,
) -> jnp.ndarray:
    """Compute a Gaussian or Matern kernel matrix."""
    r2 = jnp.maximum(cdistSQ(x, y), 0.0)
    z2 = r2 / (length_scale**2)

    if kernel == "gaussian":
        return jnp.exp(-0.5 * z2)

    z = jnp.sqrt(z2)

    if nu == 0.5:
        return jnp.exp(-z)
    if nu == 1.5:
        a = jnp.sqrt(3.0) * z
        return (1.0 + a) * jnp.exp(-a)

    a = jnp.sqrt(5.0) * z
    return (1.0 + a + a**2 / 3.0) * jnp.exp(-a)


def mmd_squared(
    samples: ArrayLike,
    reference_samples: ArrayLike,
    kernel: str = "gaussian",
    length_scale: float = 1.0,
    nu: float = 1.5,
) -> jnp.ndarray:
    """Compute the unbiased estimator of squared maximum mean discrepancy.

    Args:
        samples: Generated samples of shape (N, d).
        reference_samples: Independent reference samples of shape (M, d).
        kernel: Kernel family, either "gaussian" or "matern".
        length_scale: Positive kernel length scale.
        nu: Matern smoothness parameter, one of 0.5, 1.5, or 2.5.

    Returns:
        Scalar unbiased estimate of squared MMD. The estimate may be
        negative because of finite-sample variability.

    Raises:
        ValueError: If the inputs or kernel parameters are invalid.
    """
    x = jnp.asarray(samples)
    y = jnp.asarray(reference_samples)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Sample collections must have shape (N, d).")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Sample collections must have matching dimensions.")
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError("Each collection must contain at least two samples.")
    if length_scale <= 0:
        raise ValueError("length_scale must be positive.")
    if kernel not in ("gaussian", "matern"):
        raise ValueError("Unsupported kernel.")
    if kernel == "matern" and nu not in (0.5, 1.5, 2.5):
        raise ValueError("Matern nu must be 0.5, 1.5, or 2.5.")

    n, m = x.shape[0], y.shape[0]

    kxx = _kernel_matrix(x, x, kernel, length_scale, nu)
    kyy = _kernel_matrix(y, y, kernel, length_scale, nu)
    kxy = _kernel_matrix(x, y, kernel, length_scale, nu)

    xx = (jnp.sum(kxx) - jnp.trace(kxx)) / (n * (n - 1))
    yy = (jnp.sum(kyy) - jnp.trace(kyy)) / (m * (m - 1))
    xy = jnp.mean(kxy)

    return xx + yy - 2.0 * xy

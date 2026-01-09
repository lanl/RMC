# -*- coding: utf-8 -*-

"""Definitions of kernel functions."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.jit
def cdist(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Compute Euclidean distance between each pair of the two collections of inputs.

    Args:
        x: An m_x by d array of m_x original observations in an d-dimensional space.
        y: An m_y by d array of m_y original observations in an d-dimensional space.

    Returns:
        A m_x by m_y distance matrix. For each i and j, the metric dist(u=x[i], v=y[j]) is computed and stored in the ij-th entry.
    """
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


@jax.jit
def cdistSQ(x: ArrayLike, y: ArrayLike) -> ArrayLike:
    """Compute squared Euclidean distance between each pair of the two collections of inputs.

    Args:
        x: An m_x by d array of m_x original observations in an d-dimensional space.
        y: An m_y by d array of m_y original observations in an d-dimensional space.

    Returns:
        A m_x by m_y squared distance matrix. For each i and j, the metric dist(u=x[i], v=y[j]) is computed and stored in the ij-th entry.
    """
    return jnp.sum((x[:, None] - y[None, :]) ** 2, -1)


def RBF_Gramm(x: ArrayLike, h: float = -1) -> ArrayLike:
    """Compute Gramm matrix for RBF kernel.

    Args:
        x: An m_x by d array of m_x original observations in an d-dimensional space.
        h: Variance of the RBF kernel. By default (negative argument)
            the median distance is used as the variance.

    Returns:
        A m_x by m_x matrix of kernel evaluation between all pairs that can be obtained in the input collection.
    """

    pairwise_dists = cdistSQ(x, x)
    if h < 0:  # if h < 0, using median trick
        h = jnp.median(pairwise_dists)
        h = 0.5 * h / jnp.log(x.shape[0] + 1)

    # compute the rbf kernel
    Kxy = jnp.exp(-pairwise_dists / h / 2)

    return Kxy, h


def RBF(xc: ArrayLike, h: float, x: ArrayLike) -> float:
    """Definition of radial basis function (RBF) kernel.

    Args:
        xc: Center of RBF.
        h: Variance of RBF.
        x: point to evaluate kernel.

    Returns:
        Scalar corresponding to the kernel evaluation.
    """
    distSQ = jnp.sum((x - xc) ** 2)
    return jnp.exp(-distSQ / h)

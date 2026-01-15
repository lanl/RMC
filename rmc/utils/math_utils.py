# -*- coding: utf-8 -*-

"""Auxiliary computations for Neural Networks."""

from typing import Callable

import jax
import jax.numpy as jnp


# Taken from: https://github.com/jax-ml/jax/issues/3022
def divergence_key(f: Callable, n: int, gaussian: bool) -> Callable:
    """
    Compute the divergence of a vector field using JAX.

    Args:
        f: The vector field function R^n -> R^n.
        n: Mode of divergence computation. -1 for exact trace,
           0 for efficient exact,  and positive integers for
           stochastic estimation using Hutchinson's trace estimator.
        gaussian: Flag to use Gaussian (True) or Rademacher (False)
                  vectors for stochastic estimation.

    Returns:
        A function that computes the divergence at a point.
    """
    # Exact calculation using the trace of the Jacobian
    if n == -1:
        return jax.jit(lambda x, key: jnp.trace(jax.jacobian(f)(x)))

    # Efficient exact calculation using gradients
    if n == 0:

        def div(x, key):
            fi = lambda i, *y: f(jnp.stack(y))[i]
            dfidxi = lambda i, y: jax.grad(fi, argnums=i + 1)(i, *y)
            return sum(dfidxi(i, x) for i in range(x.shape[0]))

        return jax.jit(div)

    # Hutchinson's trace estimator for stochastic estimation
    if n > 0:

        def div(x, key):
            def vJv(key):
                _, vjp = jax.vjp(f, x)
                v = (
                    jax.random.normal(key, x.shape, dtype=x.dtype)
                    if gaussian
                    else jax.random.rademacher(key, x.shape, dtype=x.dtype)
                )
                return jnp.dot(vjp(v)[0], v)

            return jax.vmap(vJv)(jax.random.split(key, n)).mean()

        return jax.jit(div)


def divergence_2(f: Callable) -> Callable:
    """
    Compute the divergence of a vector field using JAX.

    Args:
        f: The vector field function R^n -> R^n.

    Returns:
        A function that computes the divergence at a point.
    """

    # Exact calculation using the trace of the Jacobian
    def div(x):
        fi = lambda i, *y: f(jnp.stack(y))[i]
        dfidxi = lambda i, y: jax.grad(fi, argnums=i + 1)(i, *y)
        return sum(dfidxi(i, x) for i in range(x.shape[0]))

    return jax.jit(div)


def divergence(f: Callable) -> Callable:
    """
    Compute the divergence of a vector field using JAX autograd.

    Args:
        f: The vector field function R^n -> R^n.

    Returns:
        A function for computing the divergence at a given point.
    """
    return jax.jit(lambda x: jnp.trace(jax.jacobian(f)(x)))

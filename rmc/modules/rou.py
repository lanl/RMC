# -*- coding: utf-8 -*-

"""Utilities for deploying a Reverse Ornstein--Uhlenbeck (ROU) Sampler."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.models import NN_with_time, NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict


class ReverseOUSampler(nnx.Module):
    """Definition of Reverse Ornstein--Uhlenbeck (ROU) Sampler class."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        h: float,
        T: int,
        a_schedule: Callable,
        sigma_schedule: Callable,
        verbose: bool = False,
    ):
        """Initialization of Reverse Ornstein--Uhlenbeck Sampler class.

        The noising-time reference process is

            dY_s = -a(s) Y_s ds + sqrt(2) sigma(s) dW_s,

        with generative time t = TT - s and TT = h * T.

        The schedules are represented as piecewise constant on the numerical
        time grid, with values evaluated at interval midpoints. The Gaussian
        OU transition on each interval is then evaluated exactly.

        Args:
            config: Dictionary with neural-network configuration parameters.
            densitycl: Density class representing the target density.
            h: Time step.
            T: Number of time steps.
            a_schedule: Scalar OU drift schedule a(s).
            sigma_schedule: Scalar OU diffusion schedule sigma(s).
            verbose: Verbosity flag.
        """
        super().__init__()

        self.config = config
        self.Dcl = densitycl
        self.d = config["dim"]

        self.h = h
        self.hsqrt = jnp.sqrt(h)
        self.T = T
        self.TT = h * T

        self.a_schedule = a_schedule
        self.sigma_schedule = sigma_schedule
        self.verbose = verbose

        # Evaluate the ROU schedules at noising-time interval midpoints.
        self.s_mid = (jnp.arange(T) + 0.5) * h
        self.a_ref = jax.vmap(a_schedule)(self.s_mid)
        self.sigma_ref = jax.vmap(sigma_schedule)(self.s_mid)

        # Exact Gaussian transition coefficients for a piecewise-constant
        # OU process on each noising-time interval.
        self.ref_mean, self.ref_variance = self._build_reference_transitions()

        # Cumulative conditional moments for
        #
        #   Y_s | Y_0 = y ~ N(m(s) y, c(s) I).
        #
        self.noising_mean, self.noising_variance = self._build_conditional_moments()

        # Generative time t = TT - s.  The conditional arrays have T+1
        # entries corresponding to t_k = k h.
        self.cond_mean = self.noising_mean[::-1]
        self.cond_variance = self.noising_variance[::-1]

        # Coefficients used on each generative-time interval [t_k, t_{k+1}].
        self.a_gen = self.a_ref[::-1]
        self.sigma_gen = self.sigma_ref[::-1]

        # Create NN model for the residual drift.
        #
        # The complete learned drift will be
        #
        #   f_theta(x,t) = a(TT-t) x + g_theta(x,t).
        #
        # Score-informed networks are intentionally excluded because ROU
        # only requires pointwise evaluation of the target density.
        if config["nn_type"] == "time_embed":
            self.nnmodel = NN_with_time_embedding(self.config)
        elif config["nn_type"] == "score":
            raise ValueError("ROU does not use target-gradient-informed neural networks")
        else:
            self.nnmodel = NN_with_time(self.config)

    def _build_reference_transitions(self):
        """Build the Gaussian ROU transition on each noising interval."""
        a = self.a_ref
        sigma = self.sigma_ref

        mean = jnp.exp(-a * self.h)

        nonzero = jnp.abs(a) > 1.0e-12
        a_safe = jnp.where(nonzero, a, 1.0)

        variance_nonzero = (
            sigma**2
            * (-jnp.expm1(-2.0 * a * self.h))
            / a_safe
        )
        variance_zero = 2.0 * sigma**2 * self.h

        variance = jnp.where(
            nonzero,
            variance_nonzero,
            variance_zero,
        )

        return mean, variance

    def _build_conditional_moments(self):
        """Build cumulative ROU conditional mean and variance coefficients."""

        def ou_step(carry, transition):
            mean_prev, variance_prev = carry
            mean_step, variance_step = transition

            mean_next = mean_step * mean_prev
            variance_next = mean_step**2 * variance_prev + variance_step

            return (
                mean_next,
                variance_next,
            ), (
                mean_next,
                variance_next,
            )

        initial = (
            jnp.asarray(1.0),
            jnp.asarray(0.0),
        )

        _, (mean_tail, variance_tail) = jax.lax.scan(
            ou_step,
            initial,
            (self.ref_mean, self.ref_variance),
        )

        mean = jnp.concatenate(
            (
                jnp.ones((1,)),
                mean_tail,
            )
        )
        variance = jnp.concatenate(
            (
                jnp.zeros((1,)),
                variance_tail,
            )
        )

        return mean, variance

    def eval_conditional_sample(
        self,
        x_terminal: ArrayLike,
        k: ArrayLike,
        eta: ArrayLike,
    ):
        """Sample analytically from the endpoint-conditioned ROU marginal.

        At generative grid time t_k = k h,

            X_{t_k} | X_T = y
                ~ N(m_k y, c_k I).

        Args:
            x_terminal: Terminal target endpoints.
            k: Generative-time grid indices.
            eta: Standard Gaussian random variables.

        Returns:
            Samples from the ROU conditional marginal.
        """
        mean = self.cond_mean[k]
        variance = self.cond_variance[k]

        while mean.ndim < x_terminal.ndim:
            mean = mean[..., None]
            variance = variance[..., None]

        return mean * x_terminal + jnp.sqrt(variance) * eta

    def eval_conditional_drift(
        self,
        x_terminal: ArrayLike,
        x: ArrayLike,
        k: ArrayLike,
    ):
        """Evaluate the endpoint-conditioned reverse-SDE drift.

        On generative interval k, the conditional drift is

            f(x,t;y)
              = a_bar(t) x
                - 2 sigma_bar(t)^2 / c(t) * (x - m(t) y).

        The terminal grid point k=T is excluded because c(T)=0.
        """
        mean = self.cond_mean[k]
        variance = self.cond_variance[k]
        a = self.a_gen[k]
        sigma = self.sigma_gen[k]

        while mean.ndim < x.ndim:
            mean = mean[..., None]
            variance = variance[..., None]
            a = a[..., None]
            sigma = sigma[..., None]

        return (
            a * x
            - 2.0 * sigma**2 / variance * (x - mean * x_terminal)
        )

    def eval_conditional_residual(
        self,
        eta: ArrayLike,
        k: ArrayLike,
    ):
        """Evaluate the regression target for the learned residual drift.

        For

            X_t = m(t) X_T + sqrt(c(t)) eta

        and

            f_theta(x,t) = a_bar(t) x + g_theta(x,t),

        the conditional target for the learned component is

            g_cond = -2 sigma_bar(t)^2 / sqrt(c(t)) eta.

        The expression is evaluated directly from eta to avoid cancellation
        in x - m(t) X_T near the terminal endpoint.
        """
        variance = self.cond_variance[k]
        sigma = self.sigma_gen[k]

        while variance.ndim < eta.ndim:
            variance = variance[..., None]
            sigma = sigma[..., None]

        return -2.0 * sigma**2 / jnp.sqrt(variance) * eta

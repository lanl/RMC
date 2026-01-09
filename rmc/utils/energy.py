# -*- coding: utf-8 -*-

"""Definitions for energy functions."""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

RealArray = ArrayLike


class LogDensity:
    """Base energy class for sampling from unnormalized density functions.

    A :class:`LogDensity` is the base class for all the type 2 distribution functions implemented.
    """

    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for energy
        evaluation (e.g. tempering).

        Args:
            kwargs: Additional arguments that may be used by derived classes.
        """
        raise NotImplementedError

    def log_target(self, x: RealArray) -> RealArray:
        """Definition of log-target density function.

        Args:
            x: Array of samples to evaluate log-target density.

        Returns:
            Array of evaluated log-target density.
        """
        raise NotImplementedError

    def log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of unnormalized log-posterior.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated unnormalized log-posterior.
        """
        if tempering is None:
            return self.log_target(x)
        else:  # Evaluate using tempering function
            return tempering * self.log_target(x)

    def der_log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of derivate of unnormalized log-posterior
        function.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of unnormalized log-posterior.
        """
        der_logtarget = jax.vmap(jax.grad(self.log_target))
        if tempering is None:
            return der_logtarget(x)
        else:
            return tempering * der_logtarget(x)


class LogDensityPath:
    """Base energy class for sampling from unnormalized density functions using a path adaptation.

    Corresponds to type 1 distribution function.

    A :class:`LogDensityPath` is the base class for all the type 1 energy functions implemented.
    """

    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for energy
        evaluation (e.g. tempering).

        Args:
            kwargs: Additional arguments that may be used by derived classes.
        """
        raise NotImplementedError

    def log_base(self, x: RealArray) -> RealArray:
        """Definition of log-base density function.

        Args:
            x: Array of samples to evaluate log-base density.

        Returns:
            Array of evaluated log-base density.
        """
        raise NotImplementedError

    def log_target(self, x: RealArray) -> RealArray:
        """Definition of log-target density function.

        Args:
            x: Array of samples to evaluate log-target density.

        Returns:
            Array of evaluated log-target density.
        """
        raise NotImplementedError

    def log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of unnormalized log-posterior.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated unnormalized log-posterior.
        """
        if tempering is None:
            return self.log_base(x) + self.log_target(x)
        else:  # Evaluate using tempering function
            return (1.0 - tempering) * self.log_base(x) + tempering * self.log_target(x)

    def der_log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of derivate of unnormalized log-posterior
        function.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of unnormalized log-posterior.
        """
        der_logbase = jax.vmap(jax.grad(self.log_base))
        der_logtarget = jax.vmap(jax.grad(self.log_target))
        if tempering is None:
            return der_logbase(x) + der_logtarget(x)
        else:
            return (1.0 - tempering) * der_logbase(x) + tempering * der_logtarget(x)


class LogPosterior(LogDensity):
    """Class for Bayesian posterior sampling.

    Corresponds to type 2 distribution function.

    It requires definitions for prior density function and likelihood function.
    """

    def log_prior(self, x: RealArray) -> RealArray:
        """Definition of log-prior density function.

        Args:
            x: Array of samples to evaluate log-prior.

        Returns:
            Array of evaluated log-prior.
        """
        raise NotImplementedError

    def log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of unnormalized log-posterior function.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated unnormalized log-posterior.
        """
        if tempering is None:
            return self.log_likelihood(x) + self.log_prior(x)
        else:  # Evaluate using tempering function
            return tempering * self.log_likelihood(x) + self.log_prior(x)

    def der_log_unposterior(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        """Definition of derivate of unnormalized log-posterior
        function.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of unnormalized log-posterior.
        """
        der_logprior = jax.vmap(jax.grad(self.log_prior))
        der_loglk = jax.vmap(jax.grad(self.log_likelihood))
        if tempering is None:
            return der_loglk(x) + der_logprior(x)
        else:
            return tempering * der_loglk(x) + der_logprior(x)


class LinearRegressionE(LogPosterior):
    """Functionality to evaluate energy for a linear regression
    model with fixed noise standard deviation."""

    def __init__(
        self,
        dim: int,
        data_x: ArrayLike,
        data_y: ArrayLike,
        stddev: float,
        mean_prior: ArrayLike,
        stddev_prior: ArrayLike,
    ):
        """Initialize variables and supplementary functions for energy
        evaluation.

        Args:
            dim: Dimension of regression problem.
            data_x: Array with feature (input) data.
            data_y: Array with linear model response (output) data.
            stddev: Noise standard deviation.
            mean_prior: Array with mean of the prior for the linear regression.
            stddev_prior: Array with standard deviation of the prior for the linear regression.
        """
        # Store problem definition
        self.dim = dim
        self.data_x = jnp.array(data_x)
        self.data_y = jnp.array(data_y)
        self.stddev = stddev
        self.var = stddev**2
        self.D = self.data_x.shape[0]  # Size of provided data
        # Store parameters of prior distribution
        self.mean_prior = jnp.array(mean_prior)
        self.precision_prior = jnp.diagflat(1.0 / jnp.array(stddev_prior) ** 2)
        self.log_det_prior = (
            -self.dim / 2.0 * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(stddev_prior) ** 2) / 2.0
        )

    def log_prior(self, x: RealArray) -> RealArray:
        """Definition of log-prior density function for linear
        regression model with fixed noise standard deviation.

        This corresponds to a multi-variate normal distribution
        with diagonal covariance matrix.

        Args:
            x: Array of samples to evaluate log-prior.

        Returns:
            Array of evaluated log-prior.
        """
        xvec = (x - self.mean_prior).reshape((-1, 1, x.shape[-1]))
        lp = self.log_det_prior - 0.5 * jnp.squeeze(
            xvec @ self.precision_prior @ jnp.transpose(xvec, axes=(0, 2, 1))
        )

        # print("In log_prior --> xvec.shape: ", xvec.shape)
        # print("In log_prior --> lp.shape: ", lp.shape)

        return lp

    def log_likelihood(self, x: RealArray) -> RealArray:
        """Definition of log-likelihood function for linear
        regression model with fixed noise standard deviation.

        This corresponds to a multi-variate normal distribution
        with the same constant noise standard deviation in all
        the problem dimensions.

        Args:
            x: Array of samples to evaluate log-likelihood.
        Returns:
            Array of evaluated log-likelihood.
        """
        # print("In log_likelihood --> x.shape: ", x.shape)
        # print("In log_likelihood --> self.data_x.shape: ", self.data_x.shape)

        y_eval = jnp.sum(x.reshape((-1, 1, x.shape[-1])) * self.data_x, axis=-1)
        ll = (
            -0.5 * jnp.sum((self.data_y - y_eval) ** 2 / self.var, axis=-1)
            - 0.5 * self.D * jnp.log(2 * jnp.pi)
            - 0.5 * self.D * jnp.log(self.var)
        )
        ll = ll.squeeze()

        # print("In log_likelihood --> y_eval.shape: ", y_eval.shape)
        # print("In log_likelihood --> ll.shape: ", ll.shape)
        return ll

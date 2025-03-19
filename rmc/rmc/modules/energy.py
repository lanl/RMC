# -*- coding: utf-8 -*-

"""Definitions for energy functions."""

from typing import Callable, Optional

from jax.typing import ArrayLike

import jax

import jax.numpy as jnp

RealArray = ArrayLike

class Energy:
    """Base energy class.

    A :class:`Energy` is the base class for all the energy functions implemented.
    """
    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for energy
        evaluation (e.g. tempering).

        Args:
            kwargs: Additional arguments that may be used by derived classes.
        """
        raise NotImplementedError

    def log_prior(self, x: RealArray) -> RealArray:
        """Definition of log-prior density function.

        Args:
            x: Array of samples to evaluate log-prior.

        Returns:
            Array of evaluated log-prior.
        """
        raise NotImplementedError

    def log_likelihood(self, x: RealArray) -> RealArray:
        """Definition of log-likelihood function.

        Args:
            x: Array of samples to evaluate log-likelihood.

        Returns:
            Array of evaluated log-likelihood.
        """
        raise NotImplementedError

    def log_unposterior(self, x: RealArray, step: Optional[RealArray] = None) -> RealArray:
        """Definition of unnormalized log-posterior function for
        linear regression model with fixed noise standard deviation.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            step: Time step for tempering.

        Returns:
            Array of evaluated unnormalized log-posterior.
        """
        if step is None:
            return self.log_likelihood(x) + self.log_prior(x)
        else: # Evaluate using tempering function
            return self.tempering_fn(step) * self.log_likelihood(x) + self.log_prior(x)

    def der_log_unposterior(self, x: RealArray, step: Optional[RealArray] = None) -> RealArray:
        """Definition of derivate of unnormalized log-posterior
        function for linear regression model with fixed noise
        standard deviation.

        Args:
            x: Array of samples to evaluate unnormalized log-posterior.
            step: Time step for tempering.

        Returns:
            Array of evaluated derivative of unnormalized log-posterior.
        """
        der_logprior = jax.vmap(jax.grad(self.log_prior))
        der_loglk = jax.vmap(jax.grad(self.log_likelihood))
        if step is None:
            return der_logprior(x) + der_loglk(x)
        else:
            return der_logprior(x) + self.tempering_fn(step) * der_loglk(x)


class LinearRegressionE(Energy):
    """Functionality to evaluate energy for a linear regression
    model with fixed noise standard deviation."""
    def __init__(self, dim: int, data_x: ArrayLike, data_y: ArrayLike, stddev: float, mean_prior: ArrayLike, stddev_prior: ArrayLike, tempering_fn: Optional[Callable] = None):
        """Initialize variables and supplementary functions for energy
        evaluation.

        Args:
            dim: Dimension of regression problem.
            data_x: Array with feature (input) data.
            data_y: Array with linear model response (output) data.
            stddev: Noise standard deviation.
            mean_prior: Array with mean of the prior for the linear regression.
            stddev_prior: Array with standard deviation of the prior for the linear regression.
            tempering_fn: Definition of tempering function (if applicable).
        """
        # Store problem definition
        self.dim = dim
        self.data_x = jnp.array(data_x)
        self.data_y = jnp.array(data_y)
        self.stddev = stddev
        self.var = stddev**2
        self.D = self.data_x.shape[0] # Size of provided data
        # Store parameters of prior distribution
        self.mean_prior = jnp.array(mean_prior)
        self.precision_prior = jnp.diagflat(1. /  jnp.array(stddev_prior)**2)
        self.log_det_prior = -self.dim / 2. * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(stddev_prior)**2) / 2.
        # Store tempering function
        self.tempering_fn = tempering_fn

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
        lp = self.log_det_prior - 0.5 * jnp.squeeze(xvec @ self.precision_prior @ jnp.transpose(xvec, axes=(0, 2, 1)))

        #print("In log_prior --> xvec.shape: ", xvec.shape)
        #print("In log_prior --> lp.shape: ", lp.shape)

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
        #print("In log_likelihood --> x.shape: ", x.shape)
        #print("In log_likelihood --> self.data_x.shape: ", self.data_x.shape)

        y_eval = jnp.sum(x.reshape((-1, 1, x.shape[-1])) * self.data_x, axis=-1)
        ll = -0.5 * jnp.sum((self.data_y - y_eval)**2 / self.var, axis=-1) - 0.5 * self.D * jnp.log(2 * jnp.pi) - 0.5 * self.D * jnp.log(self.var)
        ll = ll.squeeze()

        #print("In log_likelihood --> y_eval.shape: ", y_eval.shape)
        #print("In log_likelihood --> ll.shape: ", ll.shape)
        return ll

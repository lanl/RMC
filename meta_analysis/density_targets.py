#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define distributions for examples.
"""

import os

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from rmc import (
    BaseLogDensity,
    LinearRegressionDensity,
    LogDensityPath,
    PackedMultivariateNormal,
)
from io_utils import load_data

class NormND(BaseLogDensity):
    """Definition of density class for n-dimensional (nD) 
    normal distribution."""

    def __init__(self, mean: ArrayLike, cov: ArrayLike):
        """Initialization of nD normal distribution.

        Args:
            mean: Mean of the normal distribution.
            cov: Covariance matrix.
        """
        self.nD_distribution = PackedMultivariateNormal(mean, cov)


    def log_target(self, x: ArrayLike) -> ArrayLike:
        """Define log of probability for nD normal distribution.

        Args:
            x: Array to evaluate the log of target distribution.

        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        return self.nD_distribution.log_pdf(x).squeeze()


class NormMix2D(BaseLogDensity):
    """Definition of density class for mixture of 2D normal
    distributions."""

    def __init__(
        self,
        means: ArrayLike,
        sigma2: float,
        weights: ArrayLike,
        eps: float = 1e-12,
    ):
        """Initialization of mixture of normal distributions.

        Args:
            means: Means of the target normal distributions.
            sigma2: Variance per mode. Homogeneous for all the mix components.
            weights: Weights of the mix.
        """
        self.nmix = weights.shape[-1]  # Number of elements in the mix
        self.d = means.shape[-1]  # Dimension of the distribution
        self.means = means

        self.sigma2 = sigma2
        self.cov = sigma2 * jnp.eye(self.d)
        self.invcov = jnp.eye(self.d) / sigma2

        # The component covariance is sigma2 * I, so these quantities are
        # available analytically.  Avoiding a matrix inverse and determinant
        # also keeps this common example off GPU dense-linear-algebra kernels.
        logdetcov = self.d * jnp.log(sigma2)
        self.lognorm = -0.5 * (self.d * jnp.log(2.0 * jnp.pi) + logdetcov)
        self.lognmix = jnp.log(self.nmix)

        weights = weights
        self.logweights = jnp.log(weights + eps)
        self.eps = eps

    def log_target(self, x: ArrayLike) -> ArrayLike:
        """Define log of probability for mixture of 2D normal
        distributions.

        Args:
            x: Array to evaluate the log of target distribution.

        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        diff = jnp.asarray(x)[..., None, :] - self.means

        quadratic = (
            jnp.sum(
                diff**2,
                axis=-1,
            )
            / self.sigma2
        )

        component_log_density = self.logweights + self.lognorm - 0.5 * quadratic

        return (
            jax.nn.logsumexp(
                component_log_density,
                axis=-1,
            )
            - self.lognmix
        ).squeeze()


class LogisticRegDensity(LinearRegressionDensity):
    """Definition of density class for logistic regression."""

    def log_likelihood(self, x: ArrayLike) -> ArrayLike:
        """Define log likelihood for logistic regression model.

        Args:
            x: Array to evaluate the log likelihood.

        Returns:
            Log likelihood evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        eps = 1e-6
        p = jax.nn.sigmoid(jnp.sum(x.reshape((-1, 1, x.shape[-1])) * self.data_x, axis=-1))
        ll = (self.data_y * jnp.log(p + eps) + (1.0 - self.data_y) * jnp.log(1.0 - p + eps)).sum(
            axis=-1
        )
        return ll.squeeze()


class FunnelDensity(LogDensityPath):
    """Definition of density class for funnel distribution."""

    def __init__(
        self,
        dim: int,
        x0_stddev: float,
        initial_mean: ArrayLike,
        initial_cov: ArrayLike,
    ):
        """Initialization of funnel density function.

        Args:
            dim: Dimension of funnel density function.
            x0_stddev: Standard deviation of coordinate 0.
            initial_mean: Mean of initial distribution in density path.
            initial_cov: Covariance matrix of initial distribution in density path.
        """
        # Store problem definition
        self.dim = dim
        self.x0_stddev = x0_stddev
        self.x0_constant = -0.5 * jnp.log(2 * jnp.pi * x0_stddev**2)
        self.xi_constant = -(dim - 1) * 0.5 * jnp.log(2 * jnp.pi)

        # Create and store distribution for initial sampling
        assert len(initial_mean.squeeze()) == dim
        assert initial_cov.shape[-1] == dim
        self.initial = PackedMultivariateNormal(initial_mean, initial_cov)

    def log_initial(self, x: ArrayLike) -> ArrayLike:
        """Define log probability for initial density in transport path.

        Args:
            x: Array to evaluate the log probability of base distribution.

        Returns:
            Log probability of initial density evaluated at provided samples.
        """
        return self.initial.log_pdf(x).squeeze()

    def log_target(self, x: ArrayLike) -> ArrayLike:
        """Define log probability for target density in transport path.

        Args:
            x: Array to evaluate the log probability of target density.

        Returns:
            Log probability of target density evaluated at provided samples.
        """
        lt = (
            self.x0_constant
            - 0.5 * x[..., :1] ** 2 / self.x0_stddev**2
            + self.xi_constant
            - 0.5 * (self.dim - 1) * x[..., :1]
            - 0.5 * jnp.sum(x[..., 1:] ** 2 / jnp.exp(x[..., :1]), axis=-1, keepdims=True)
        )
        return lt.squeeze()


def build_Neal_funnel_density():
    """Build Neal's funnel density class.

    Returns:
        Funnel density class.
    """
    d = 10  # Dimension of funnel distribution
    x0_stddev = 3.0  # Standard deviation for component 0

    # Define initial distribution
    mean0 = 0.0  # initial mean
    std0 = 1.0  # initial standard deviation
    # Initial mean in array form
    mean0_arr = mean0 * jnp.ones((1, d))
    # Initial covariance matrix
    cov0_arr = jnp.diagflat(std0**2 * jnp.ones((d,))).reshape((1, d, d))

    return FunnelDensity(d, x0_stddev, mean0_arr, cov0_arr)


def build_logistic_regression_from_data(fname: str):
    _, dir = os.path.split(os.getcwd())
    if dir == "meta_analysis":
        path2file = f"../examples/datasets/{fname}"
    else:
        path2file = f"examples/datasets/{fname}"

    x, y = load_data(path2file)
    d = x.shape[1]  # Dimension of features
    X = jnp.array(x)
    Y = jnp.array(y)
    # define prior
    prior_mean = 0.0  # prior mean on linear regression coeficients
    prior_std = 1.0  # prior standard deviation on coeficients
    prior_mean_vec = prior_mean * jnp.ones((1, d))
    prior_std_vec = prior_std * jnp.ones((1, d))
    noise_std = 1.0
    return LogisticRegDensity(d, X, Y, noise_std, prior_mean_vec, prior_std_vec)    


def build_density_class(target: str):
    """Build density class for a given target.

    Args:
        target: Name of the target distribution.

    Returns:
        Density class for the specified target distribution.
    """
    if target == "iso_10D":
        mean = jnp.zeros((1, 10))
        cov = jnp.eye(10).reshape((1, 10, 10))
        return NormND(mean, cov)
    elif target == "aniso_10D":
        mean = jnp.zeros((1, 10))
        cov = jnp.diagflat(jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])).reshape((1, 10, 10))
        return NormND(mean, cov)
    elif target == "funnel":
        return build_Neal_funnel_density()
    elif target == "gmm_2D_9":
        d = 2
        grid = 1.0
        means = jnp.array(
            [[grid * i, grid * j] for i in range(-1, 2) for j in range(-1, 2)],
            dtype=jnp.float32,
        )
        sigma2 = 0.012
        weights = jnp.ones(means.shape[0])
        return NormMix2D(means, sigma2, weights)
    elif target == "logistic_sonar":
        # Load sonar dataset and create logistic regression density
        return build_logistic_regression_from_data("sonar_full.pkl")
    elif target == "DW-4":
        # Double well potential in 4D
        mean = jnp.zeros((1, 4))
        cov = jnp.eye(4).reshape((1, 4, 4))
        return NormND(mean, cov)  # Placeholder for actual DW-4 implementation
    elif target == "LJ-13":
        # Lennard-Jones potential in 13D
        mean = jnp.zeros((1, 13))
        cov = jnp.eye(13).reshape((1, 13, 13))
        return NormND(mean, cov)  # Placeholder for actual LJ-13 implementation
    else:
        raise ValueError
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


@jax.jit
def dw4_energy(x: ArrayLike, a: float = 0.0, b: float =-4.0, c: float = 0.9, 
               d0: float = 4.0, tau: float = 1.0):
    """
    Double-well energy for a system of particles (DW-4 for N=4, 2D).

    Args:
        x: array of shape (N, 2) or flat (2N,) with particle coordinates.
        a, b, c, d0: double-well parameters.
        tau: temperature-like scaling (energy is divided by tau).

    Returns:
        Scalar energy.
    """
    x = x.reshape(-1, 2)
    n = x.shape[0]

    diff = x[:, None, :] - x[None, :, :]        # (N, N, 2)
    r2 = jnp.sum(diff ** 2, axis=-1)            # (N, N)

    # Safe distance: avoid sqrt(0) on the diagonal (NaN gradients)
    eye = jnp.eye(n, dtype=bool)
    d = jnp.sqrt(jnp.where(eye, 1.0, r2))

    u = d - d0
    pair = a * u + b * u ** 2 + c * u ** 4
    pair = jnp.where(eye, 0.0, pair)

    # Full matrix counts each pair twice -> 0.5 factor
    return 0.5 * jnp.sum(pair) / tau


@jax.jit
def lj_energy(positions: ArrayLike, epsilon: float = 1.0, sigma: float = 1.0):
    """
    Total Lennard-Jones energy of a cluster.

    Args:
        positions: (N, 3) array of atomic coordinates (N=13 for LJ-13).
        epsilon: well depth.
        sigma: distance at which the potential is zero.

    Returns:
        Scalar total potential energy.
    """
    n = positions.shape[0]

    # Pairwise displacement and squared distances
    diff = positions[:, None, :] - positions[None, :, :]   # (N, N, 3)
    r2 = jnp.sum(diff ** 2, axis=-1)                       # (N, N)

    # Mask the diagonal (i == j) to avoid division by zero.
    # Using the "double where" trick keeps gradients NaN-free.
    eye = jnp.eye(n, dtype=bool)
    r2_safe = jnp.where(eye, 1.0, r2)

    inv_r2 = (sigma ** 2) / r2_safe
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2

    pair_energy = 4.0 * epsilon * (inv_r12 - inv_r6)
    pair_energy = jnp.where(eye, 0.0, pair_energy)

    # Each pair is counted twice in the full matrix
    return 0.5 * jnp.sum(pair_energy)


class DW_4(BaseLogDensity):
    """Definition of density class for Double Well (DW) potential
    with 4 particles."""
    def __init__(self, a: float, b: float, c: float, d0: float, tau: float):
            """Initialization of DW-4 parameters.
    
            Args:
                epsilon: well depth.
                sigma: distance at which the potential is zero.
            """
            self.a = a
            self.b = b
            self.c = c
            self.d0 = d0
            self.tau = tau
    
    
    def log_target(self, x: ArrayLike) -> ArrayLike:
        """Define log of probability for LJ-13 potential.
    
        Args:
            x: Array to evaluate the log of target distribution.
    
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        # jax.jit(jax.vmap(
        return -dw4_energy(x, self.a, self.b, self.c, self.d0, self.tau)


class LJ_13(BaseLogDensity):
    """Definition of density class for Lennard-Jones potential
    with 13 particles."""
    def __init__(self, epsilon: float, sigma: float):
            """Initialization of LJ-13 parameters.
    
            Args:
                epsilon: well depth.
                sigma: distance at which the potential is zero.
            """
            self.epsilon = epsilon
            self.sigma = sigma
    
    
    def log_target(self, x: ArrayLike) -> ArrayLike:
        """Define log of probability for LJ-13 potential.
    
        Args:
            x: Array to evaluate the log of target distribution.
    
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        # jax.jit(jax.vmap(-lj_energy(x, self.epsilon, self.sigma)))
        return -lj_energy(x, self.epsilon, self.sigma)


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
        # Double well potential for 4 particles
        a = 0.0
        b =-4.0 
        c = 0.9
        d0 = 4.0
        tau = 1.0
        return DW_4(a, b, c, d0, tau)
    elif target == "LJ-13":
        # Lennard-Jones potential for 13 particles
        epsilon = 1.0
        sigma = 1.0
        return LJ_13(epsilon, sigma)
    else:
        raise ValueError
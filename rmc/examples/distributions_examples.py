#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define distributions for examples.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from rmc import LinearRegressionE, LogDensity, LogDensityPath

RealArray = ArrayLike


class ENorm2D(LogDensity):
    """Definition of energy class for a 2D normal distribution."""
    def __init__(self, cov: RealArray):
        """Initialization of 2D normal distribution.
        
        Args:
            cov: Covariance matrix.
        """
        self.cov = cov
        self.invcov = jnp.linalg.inv(cov)

    def log_target(self, x: RealArray) -> RealArray:
        """Define log of probability for 2D normal distribution.
        
        Args:
            x: Array to evaluate the log of target distribution.
        
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        ll = -0.5 * x @ self.invcov @ x.T
        return ll.squeeze()


class ENormMix2D(LogDensity):
    """Definition of energy class for mixture of 2D normal 
    distributions."""
    def __init__(self, means: RealArray, sigma2: float, weights: RealArray, eps: float=1e-12):
        """Initialization of mixture of normal distributions.
        
        Args:
            means: Means of the normal distributions.
            cov: Covariance matrix. Homogeneous for all the mix components.
            weights: Weights of the mix.
        """
        self.nmix = weights.shape[-1] # Number of elements in the mix
        #assert means.shape[1] == self.nmix
        #assert means.shape[-1] == cov.shape[-1]
        self.d = means.shape[-1] # Dimension of the distribution
        self.means = means
        
        self.sigma2 = sigma2
        self.cov = sigma2 * jnp.eye(self.d) # variance converted to covariance matrix
        
        self.invcov = jnp.linalg.inv(self.cov)
        logdetcov = jnp.linalg.det(self.cov)
        self.lognorm = -0.5 * (self.d * jnp.log(2. * jnp.pi) + logdetcov)
        
        weights = weights / weights.sum()
        self.logweights = jnp.log(weights + eps)

    def log_target(self, x: RealArray) -> RealArray:
        """Define log of probability for mixture of 2D normal 
        distributions.
        
        Args:
            x: Array to evaluate the log of target distribution.
        
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        diff = x.reshape((-1, 1, self.d)) - self.means # 3D array: B nmix d
        ll = self.lognorm - 0.5 * jnp.einsum("...kd,...de,...ke->...k", diff, self.invcov, diff)
        #ll = self.lognorm - 0.5 * jnp.sum(diff @ self.invcov @ jnp.transpose(diff, axes=(0, 2, 1)), axis=-1)
        rr = jnp.sum(jnp.logaddexp(self.logweights, ll), axis=-1)
        return rr.squeeze()


class ENormMix2D_(LogDensity):
    """Definition of energy class for mixture of 2D normal 
    distributions."""
    def __init__(self, means: RealArray, sigma2: float, weights: RealArray, eps: float=1e-12):
        """Initialization of mixture of normal distributions.
        
        Args:
            means: Means of the normal distributions.
            sigma2: Variance of component. Homogeneous for all the mix components.
            weights: Weights of the mix.
        """
        self.nmix = weights.shape[-1] # Number of elements in the mix
        #assert means.shape[1] == self.nmix
        #assert means.shape[-1] == cov.shape[-1]
        self.d = means.shape[-1] # Dimension of the distribution
        self.means = means
        self.means = means.squeeze()
        
        self.sigma2 = sigma2
        self.cov = sigma2 * jnp.eye(self.d) # variance converted to covariance matrix
        self.invcov = jnp.linalg.inv(self.cov)
        logdetcov = jnp.linalg.det(self.cov)
        self.lognorm = -0.5 * (self.d * jnp.log(2. * jnp.pi) + logdetcov)
        
        self.weights = weights / weights.sum()
        self.logweights = jnp.log(self.weights + eps)
        self.eps = eps
        
    def mvnpdfsum(self, x, i, psum):
        """Compute probability density function for a mutivariate normal
        distribution centered at one location of the skeleton.
        
        Args:
            x: Array to evaluate the log of target distribution.
            i: Index of the mean of the normal distribution to evaluate.
            psum: Current accumulation of evaluated probability densities.
        """
        #dd = self.cov # jnp.eye(self.d) * self.cov[0,0]**2
        #print("dd shape: ", dd.shape)
        #print("means shape: ", self.means.shape)
        #print("x shape: ", x.shape)
        #dd2 = jax.scipy.stats.multivariate_normal.pdf(x, mean=self.means[i, :], cov=jnp.eye(self.d) * self.cov[0,0]**2)
        #print("dd2 shape: ", dd2.shape)
        #print("psum shape: ", psum.shape)
        #print("self.logweights[i] shape: ", self.logweights[i].shape)
        
        return psum + jnp.exp(self.logweights[i] + jax.scipy.stats.multivariate_normal.pdf(x, mean=self.means[i, :], cov=self.cov))
        
        #return psum + jax.scipy.stats.multivariate_normal.pdf(x, mean=self.means[i, :], cov=self.cov)

    def log_target(self, x: RealArray) -> RealArray:
        """Define log of probability for mixture of 2D normal 
        distributions.
        
        Args:
            x: Array to evaluate the log of target distribution.
        
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        
        #funcbody = partial(self.mvnpdfsum, jnp.ravel(x))
        #p = jax.lax.fori_loop(0, self.nmix, funcbody, 0.)
        if x.ndim > 1:
            sum0 = self.eps * jnp.ones(x.shape[0])
        else:
            sum0 = self.eps # 0.
        funcbody = partial(self.mvnpdfsum, x)
        p = jax.lax.fori_loop(0, self.nmix, funcbody, sum0)
        ll = jnp.log(p) - jnp.log(self.nmix)
        #print("ll shape: ", ll.shape)
        return ll.squeeze()

class ESkeleton2D(LogDensity):
    """Definition of energy class for mix of equal weight 2D normal 
    distributions with centers delineating a specified skeleton.
    """
    def __init__(self, z: RealArray, sigma: float):
        """Initialization of 2D skeleton distribution.
        
        Args:
            z: Means of the normal distributions that delineate the skeleton.
            sigma: Standard deviation. Homogeneous for all the components.
        """
        self.z = jnp.array(z)
        self.sigma = sigma
        self.numd, self.dim = self.z.shape

    def mvnpdfsum(self, x, i, psum):
        """Compute probability density function for a mutivariate normal
        distribution centered at one location of the skeleton.
        
        Args:
            x: Array to evaluate the log of target distribution.
            i: Index of the mean of the normal distribution to evaluate.
            psum: Current accumulation of evaluated probability densities.
        """
        return psum + jax.scipy.stats.multivariate_normal.pdf(x, mean=self.z[i, :], cov=jnp.eye(self.dim) * self.sigma**2)

    def log_target(self, x: RealArray) -> RealArray:
        """Define log of probability of 2D skeleton distribution.
        
        Args:
            x: Array to evaluate the log of target distribution.
        
        Returns:
            Log of target distribution evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        funcbody = partial(self.mvnpdfsum, jnp.ravel(x))
        p = jax.lax.fori_loop(0, self.numd, funcbody, 0.)
        ll = jnp.log(p) - jnp.log(self.numd)
        return ll.squeeze()


class ELogisticReg(LinearRegressionE):
    """Definition of energy class for logistic regression."""
    def log_likelihood(self, x: RealArray) -> RealArray:
        """Define log likelihood for logistic regression model.
        
        Args:
            x: Array to evaluate the log likelihood.
        
        Returns:
            Log likelihood evaluated at provided samples.
            (squeeze needed for auto grad operations.)
        """
        eps = 1e-6
        p = jax.nn.sigmoid(jnp.sum(x.reshape((-1, 1, x.shape[-1])) * self.data_x, axis=-1))
        ll = (self.data_y * jnp.log(p + eps) + (1. - self.data_y) * jnp.log(1. - p + eps)).sum(axis=-1)
        return ll.squeeze()


class FunnelE(LogDensityPath):
    """Definition of energy class for funnel distribution."""
    def __init__(self, dim: int, x0_stddev: float, mean_others: ArrayLike, stddev_others: ArrayLike,):
        """Initialization of funnel distribution.
        
        Args:
            dim: Dimension of funnel distribution.
            x0_stddev: Standard deviation of coordinate 0.
            mean_others: Mean of coordinates > 0. 
            stddev_others: Standard deviation of coordinates > 0.
        """
        # Store problem definition
        self.dim = dim
        self.x0_stddev = x0_stddev
        self.x0_constant = -0.5 * jnp.log(2 * jnp.pi * x0_stddev**2)
        self.xi_constant = -(dim - 1) * 0.5 * jnp.log(2 * jnp.pi)
        # Store parameters for dimensions > 0
        self.mean_others = jnp.array(mean_others)
        self.precision_others = jnp.diagflat(1. /  jnp.array(stddev_others)**2)
        self.log_det_others = -self.dim / 2. * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(stddev_others)**2) / 2.
        
    def log_base(self, x: RealArray) -> RealArray:
        """Define log probability for base distribution in transport path.
        
        Args:
            x: Array to evaluate the log probability of base distribution.
        
        Returns:
            Log probability of base distribution evaluated at provided samples.
        """
        xvec = (x - self.mean_others).reshape((-1, 1, x.shape[-1]))
        lb = self.log_det_others - 0.5 * jnp.squeeze(xvec @ self.precision_others @ jnp.transpose(xvec, axes=(0, 2, 1)))
        return lb.squeeze()
        
    def log_target(self, x: RealArray) -> RealArray:
        """Define log probability for target distribution in transport path.
        
        Args:
            x: Array to evaluate the log probability of target distribution.
        
        Returns:
            Log probability of target distribution evaluated at provided samples.
        """
        lt = self.x0_constant - 0.5 * x[..., :1]**2 / self.x0_stddev**2 + self.xi_constant - 0.5 * (self.dim - 1) * x[..., :1]  - 0.5 * jnp.sum(x[..., 1:]**2 / jnp.exp(x[..., :1]), axis=-1, keepdims=True)
        return lt.squeeze()

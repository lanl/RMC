# -*- coding: utf-8 -*-

"""Utilities for sampling and getting statistics from common distributions"""

from abc import ABC, abstractmethod
from typing import Tuple

import jax
from jax.scipy.stats import multivariate_normal, norm, uniform
from jax.typing import ArrayLike


class BasePackedDistribution(ABC):
    """Base distribution class for sampling and getting statistics.

    A :class:`BasePackedDistribution` is the base class for packing
    existing distributions and enable a dual role of sampling and
    computing statistics (e.g. log probability density function).
    """

    # Flag to indicate that one can sample from this class
    can_sample: bool = True
    # Flag to indicate that one can evaluate log pdf from this class
    has_logpdf: bool = True

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for
        distribution sampling and statistics evaluation (e.g.
        logpdf).

        Args:
            kwargs: Additional arguments that may be used by derivated
                classes.
        """
        raise NotImplementedError

    @abstractmethod
    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        """Computation of the log probability density function (pdf).

        Args:
            x: Array of samples to evaluate log pdf.

        Returns:
            Array of evaluated log pdf.
        """
        raise NotImplementedError

    @abstractmethod
    def rvs(self, shape: Tuple[int], key: ArrayLike) -> ArrayLike:
        """Random variable sampling (rvs).

        Args:
            shape: Tuple describing the shape of samples to generate.
            key: Key for random number generation.

        Returns:
            Generated samples.
        """
        raise NotImplementedError


class PackedNormal(BasePackedDistribution):
    """Class for sampling and getting statistics from univariate normal
    distribution."""

    def __init__(self, mean: float = 0.0, stddev: float = 1.0):
        """Set parameters of univariate normal distribution.

        Args:
            mean: Mean value of the univariate normal distribution.
            stddev: Standard deviation of the univariate normal distribution.
        """
        self.mean = mean
        self.stddev = stddev
        self.loc = mean  # Distribution offset parameter (jax scipy)
        self.scale = stddev  # Distribution scale parameter (jax scipy)

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        """Computation of the log pdf for univariate normal distribution.

        Args:
            x: Array of samples to evaluate log pdf.

        Returns:
            Array of evaluated univariate normal log pdf.
        """
        return norm.logpdf(x, loc=self.loc, scale=self.scale)

    def rvs(self, shape: Tuple[int], key: ArrayLike) -> ArrayLike:
        """Random variable sampling (rvs) from univariate normal distribution.

        Args:
            shape: Tuple describing the shape of samples to generate.
            key: Key for random number generation.

        Returns:
            Generated univariate normal distribution samples.
        """
        return jax.random.normal(key, shape) * self.stddev + self.mean


class PackedMultivariateNormal(BasePackedDistribution):
    """Class for sampling and getting statistics from multivariate normal
    distribution."""

    def __init__(self, mean: ArrayLike, cov: ArrayLike):
        """Set parameters of multivariate normal distribution.

        Args:
            mean: Mean value of the multivariate normal distribution.
            cov: Covariance matrix of the multivariate normal distribution.
        """
        self.mean = mean
        self.cov = cov

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        """Computation of the log pdf for multivariate normal distribution.

        Args:
            x: Array of samples to evaluate log pdf.

        Returns:
            Array of evaluated multivariate normal log pdf.
        """
        return multivariate_normal.logpdf(x, mean=self.mean, cov=self.cov)

    def rvs(self, shape: Tuple[int], key: ArrayLike) -> ArrayLike:
        """Random variable sampling (rvs) from univariate normal distribution.

        Args:
            shape: Tuple describing the shape of samples to generate.
            key: Key for random number generation.

        Returns:
            Generated univariate normal distribution samples.
        """
        return jax.random.multivariate_normal(key, self.mean, self.cov, shape)


class PackedUniform(BasePackedDistribution):
    """Class for sampling and getting statistics from uniform distribution."""

    def __init__(self, minval: float = 0.0, maxval: float = 1.0):
        """Set parameters of uniform distribution.

        Args:
            minval: Inclusive minimum value to generate in the samples of
                the uniform distribution.
            maxval: Exclusive maximum value to generate in the samples of
                the uniform distribution.
        """
        self.minval = minval
        self.maxval = maxval
        self.loc = minval  # Distribution offset parameter (jax scipy)
        self.scale = maxval - minval  # Distribution scale parameter (jax scipy)

    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        """Computation of the log pdf for uniform distribution.

        Args:
            x: Array of samples to evaluate uniform log pdf.

        Returns:
            Array of evaluated uniform log pdf.
        """
        return uniform.logpdf(x, loc=self.loc, scale=self.scale)

    def rvs(self, shape: Tuple[int], key: ArrayLike) -> ArrayLike:
        """Random variable sampling (rvs) from uniform distribution.

        Args:
            shape: Tuple describing the shape of samples to generate.
            key: Key for random number generation.

        Returns:
            Generated uniform distribution samples.
        """
        return jax.random.uniform(key, shape, minval=self.minval, maxval=self.maxval)

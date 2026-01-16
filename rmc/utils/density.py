# -*- coding: utf-8 -*-

"""Definitions of density functions to sample from."""

from abc import ABC, abstractmethod
from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

RealArray = ArrayLike


class BaseLogDensity(ABC):
    """Base density class for sampling from unnormalized density functions.

    A :class:`BaseLogDensity` is the base class for constructing density functions
    to sample from.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for density
        evaluation (e.g. tempering).

        Args:
            kwargs: Additional arguments that may be used by derived classes.
        """

    @abstractmethod
    def log_target(self, x: RealArray) -> RealArray:
        """Definition of log-target density function, i.e. the unnormalized
        probability density function that is hard-to-sample.

        The unnormalized target density function is denoted as :math:`\tilde{\nu}(x)`.

        Args:
            x: Array of samples to evaluate log-target density.

        Returns:
            Array of evaluated log-target density.
        """

    def log_target_proposal(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        r"""Definition of the logarithm of the target proposal density function to
        use in the sampling method.

        The target proposal may correspond to a tempered density, a time dependent
        target density (usually denoted as type 1 density in the literature
        :cite:`tian-2024-lfis`) or a posterior density in Bayesian sampling (usually
        denoted as type 2 density in the literature :cite:`tian-2024-lfis`).

        Args:
            x: Array of samples to evaluate target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated log of the target proposal.
        """
        if tempering is None:
            return self.log_target(x)
        else:  # Evaluate using tempering function
            return tempering * self.log_target(x)

    def der_log_target_proposal(
        self, x: RealArray, tempering: Optional[RealArray] = None
    ) -> RealArray:
        """Definition of the derivative of the logarithm of the target proposal
        density function to use in the sampling method.

        Args:
            x: Array of samples to evaluate derivative of log-target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of log-target proposal.
        """
        der_logtarget = jax.vmap(jax.grad(self.log_target))
        if tempering is None:
            return der_logtarget(x)
        else:
            return tempering * der_logtarget(x)


class LogDensityPath(BaseLogDensity):
    r"""Abstract class defining methods for sampling from type 1 unnormalized
    density functions.

    The type 1 density function :cite:`tian-2024-lfis` is a time dependent
    target density function :math:`\tilde{\rho}_{*}(x, t)` representing the
    gradual deformation of an easy-to-sample initial density function :math:`\mu`
    into a target hard-to-sample density function :math:`\nu`. This is defined as

    .. math:: \tilde{\rho}_{*}(x, t) = \mu^{1 - \tau(t)}(x) \; \tilde{\nu}^{\tau(t)}(x) \;,

    with :math:`\tau(t)` a schedule function monotonically transforming time
    :math:`t`.
    """

    @abstractmethod
    def log_initial(self, x: RealArray) -> RealArray:
        r"""Definition of log-initial density function.

        This is an an easy-to-sample initial density function denoted as
        :math:`\mu(x)`.

        Args:
            x: Array of samples to evaluate log-initial density.

        Returns:
            Array of evaluated log-base density.
        """

    def log_target_proposal(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        r"""Definition of the logarithm of the type 1 target proposal density
        function to use in the sampling method.

        The target proposal corresponds to :math:`\tilde{\rho}_{*}(x, t)`.

        Args:
            x: Array of samples to evaluate target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated type 1 log-target density proposal function.
        """
        if tempering is None:
            return self.log_initial(x) + self.log_target(x)
        else:  # Evaluate using tempering function
            return (1.0 - tempering) * self.log_initial(x) + tempering * self.log_target(x)

    def der_log_target_proposal(
        self, x: RealArray, tempering: Optional[RealArray] = None
    ) -> RealArray:
        """Definition of the derivative of the logarithm of the type 1 target
        proposal density function to use in the sampling method.

        Args:
            x: Array of samples to evaluate derivative of log-target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of type 1 log-target proposal function.
        """
        der_loginitial = jax.vmap(jax.grad(self.log_initial))
        der_logtarget = jax.vmap(jax.grad(self.log_target))
        if tempering is None:
            return der_loginitial(x) + der_logtarget(x)
        else:
            return (1.0 - tempering) * der_loginitial(x) + tempering * der_logtarget(x)


class LogDensityPosterior(BaseLogDensity):
    r"""Abstract class defining methods for sampling from type 2 unnormalized
    density functions.

    The type 2 density function :cite:`tian-2024-lfis` is a time dependent
    target density function :math:`\tilde{\rho}_{*}(x, t)` representing the
    posterior density in Bayesian sampling where a prior density function
    :math:`\pi` and a likelihood density function :math:`L` are given. This
    is defined as

    .. math:: \tilde{\rho}_{*}(x, t) = L^{\tau(t)}(x) \; \pi(x)  \;,

    with :math:`\tau(t)` a schedule function monotonically transforming time
    :math:`t`.
    """

    @abstractmethod
    def log_prior(self, x: RealArray) -> RealArray:
        """Definition of log-prior density function.

        The prior corresponds to :math:`\pi(x)`.

        Args:
            x: Array of samples to evaluate log-prior.

        Returns:
            Array of evaluated log-prior.
        """

    @abstractmethod
    def log_likelihood(self, x: RealArray) -> RealArray:
        """Definition of log-likelihood density function.

        The likelihood corresponds to :math:`L(x)`.

        Args:
            x: Array of samples to evaluate log-likelihood.

        Returns:
            Array of evaluated log-likelihood.
        """

    def log_target_proposal(self, x: RealArray, tempering: Optional[RealArray] = None) -> RealArray:
        r"""Definition of the logarithm of the type 2 target proposal density
        function to use in the sampling method.

        The target proposal corresponds to :math:`\tilde{\rho}_{*}(x, t)`.

        Args:
            x: Array of samples to evaluate target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated type 2 log-target density proposal function.
        """
        if tempering is None:
            return self.log_likelihood(x) + self.log_prior(x)
        else:  # Evaluate using tempering function
            return tempering * self.log_likelihood(x) + self.log_prior(x)

    def der_log_target_proposal(
        self, x: RealArray, tempering: Optional[RealArray] = None
    ) -> RealArray:
        """Definition of the derivative of the logarithm of the type 2 target
        proposal density function to use in the sampling method.

        Args:
            x: Array of samples to evaluate derivative of log-target proposal.
            tempering: Tempering factor (if used).

        Returns:
            Array of evaluated derivative of type 2 log-target proposal function.
        """
        der_logprior = jax.vmap(jax.grad(self.log_prior))
        der_loglk = jax.vmap(jax.grad(self.log_likelihood))
        if tempering is None:
            return der_loglk(x) + der_logprior(x)
        else:
            return tempering * der_loglk(x) + der_logprior(x)


class LinearRegressionDensity(LogDensityPosterior):
    """Class to evaluate logs for a linear regression
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
        y_eval = jnp.sum(x.reshape((-1, 1, x.shape[-1])) * self.data_x, axis=-1)
        ll = (
            -0.5 * jnp.sum((self.data_y - y_eval) ** 2 / self.var, axis=-1)
            - 0.5 * self.D * jnp.log(2 * jnp.pi)
            - 0.5 * self.D * jnp.log(self.var)
        )
        ll = ll.squeeze()

        return ll

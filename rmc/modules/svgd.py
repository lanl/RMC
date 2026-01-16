# -*- coding: utf-8 -*-

"""Definitions for sampler modules based on Stein variational gradient descent."""

from typing import Tuple

import jax.numpy as jnp
from jax.typing import ArrayLike

from rmc.modules.sampler import Sampler
from rmc.utils.config_dict import ConfigDict
from rmc.utils.kernels import RBF_Gramm

RealArray = ArrayLike


class SVGD(Sampler):
    """Stein Variational Gradient Descent (SVGD) sampler."""

    def __init__(self, Nsamples: int, config: ConfigDict):
        """
        Args:
            Nsamples: Number of samples to draw.
            config: Dictionary with sampler configuration parameters.
        """
        self.Nsamples = Nsamples
        super().__init__(config)
        self.maxiter = self.config["maxiter"]
        self.step_size_ = self.config["step_size"]

        # Density function
        self.D_cl = self.config["density_cl"]

        # Kernel configuraion
        # self.kernel = self.config["kernel"]
        self.kpar = self.config["kernel_parameter"]
        self.kpar0 = self.kpar
        self.alpha = self.config["update_weight"]

    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform required random state initialization."""
        self.theta = samples.copy()
        return key

    def svgd_kernel(self, particles: ArrayLike):
        """Compute Kernel matrix and corresponding derivative.

        Args:
            particles: Samples to evaluate kernel matrix and derivatives.

        Returns:
            A m_x by m_x matrix of kernel evaluation between all pairs that can be
            obtained from the samples and a m_x by m_x matrix corresponding to the
            derivative with respect to the sample positions (x).
        """
        if self.kpar0 < 0:  # and self.itnum < 2000 and self.itnum % 200 == 0:
            # If an initial parameter was not specified and
            # If low number iterations (the parameter is still adapting)
            # Then: periodically adapt the kernel parameter
            Kxy, self.kpar = RBF_Gramm(particles)
        else:
            # Do not adapt the kernel parameter
            Kxy, _ = RBF_Gramm(particles, self.kpar)

        dxkxy = -jnp.matmul(Kxy, particles)
        sumkxy = jnp.sum(Kxy, axis=1)
        for i in range(particles.shape[1]):
            dxkxy = dxkxy.at[:, i].add(jnp.multiply(particles[:, i], sumkxy))
        dxkxy = dxkxy / self.kpar
        return (Kxy, dxkxy)

    def step(
        self,
        key: ArrayLike,
        prev_samples: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Compute one step of sampler."""
        # lnpgrad = lnprob(prev_samples)
        lnpgrad = self.D_cl.der_log_target_proposal(prev_samples)
        # calculate kernel matrix and derivative
        kxy, dxkxy = self.svgd_kernel(prev_samples)
        gradx = (jnp.matmul(kxy, lnpgrad) + dxkxy) / prev_samples.shape[0]

        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        if self.itnum == 0:
            historical_grad = historical_grad + gradx**2
        else:
            historical_grad = self.alpha * historical_grad + (1 - self.alpha) * (gradx**2)

        adj_grad = jnp.divide(gradx, fudge_factor + jnp.sqrt(historical_grad))
        samples = prev_samples + self.step_size_ * adj_grad

        return key, samples

    def print_stats(self):
        """Print statistics computed during sample generation."""
        print(f"Iter: {self.itnum:>5d}, RBF variance: {self.kpar:>7.6e}")

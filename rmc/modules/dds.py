# -*- coding: utf-8 -*-

"""Utilities for deploying a Denoising Diffusion Sampler (DDS) as
demonstrated in :cite:`vargas-2023-dds`."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import save_model, train
from rmc.utils.packed_distributions import PackedMultivariateNormal

from .pis import NN_with_time, NN_with_time_embedding


class DenoisingDiffusionSampler(nnx.Module):
    """Definition of Denoising Diffusion Sampler (DDS) class."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        sigma: float,
        K: int,
        beta_schedule: Callable,
        verbose: bool = False,
    ):
        """Initialization of Denoising Diffusion Sampler class.

        Args:
            config: Dictionary with DDS configuration parameters.
            densitycl: Density class representing function to sample from.
            sigma: Standard deviation of reference process.
            K: Number of time steps.
            beta_schedule: Non-decreasing time function.
            verbose: Verbosity flag. Display configuration and steps if true.
        """
        super().__init__()

        # Store configuration
        self.config = config

        # Store density class representing target density function and components
        self.Dcl = densitycl

        # Store dimension
        self.d = config["dim"]
        # Store time steps K
        self.K = K
        # Store beta and alpha schedules
        self.beta = beta_schedule(K)
        self.alpha = 1.0 - self.beta
        # Store standard deviation and variance for reference process
        self.sigma = sigma
        self.sigmaSQ = sigma**2
        self.ref_process = PackedMultivariateNormal(
            jnp.zeros(self.d).reshape((1, self.d)),
            self.sigmaSQ * jnp.eye(self.d).reshape((1, self.d, self.d)),
        )

        # Create NN model
        if config["time_embed"]:
            self.nnmodel = NN_with_time_embedding(self.config)
        else:
            self.nnmodel = NN_with_time(self.config)

    def compute_loss(self, ddsnn: Callable, x: ArrayLike, y: ArrayLike, key: ArrayLike):
        """Evaluate cost for DDS model.

        Args:
            ddsnn: Neural network model learned for DDS.
            x: Samples from reference process.
            y: Dum variable (for compatibility with trainer).
            key: JAX random generation.

        Returns:
            Current loss.
        """

        nsamples = x.shape[0]
        # Initialize r to zero
        r = jnp.zeros(nsamples)
        # Use samples from reference process
        y = x

        for k in range(self.K):
            alpha_ = self.alpha[self.K - k - 1]
            lmbda_ = 1.0 - jnp.sqrt(1.0 - alpha_)
            key, subkey = jax.random.split(key)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            dk = float(self.K - k) / self.K
            nneval = ddsnn(y, dk)
            # nneval = ddsnn(y, self.K - k)
            y = (
                (1.0 - lmbda_) * y
                + 2.0 * self.sigmaSQ * lmbda_ * nneval
                + self.sigma * jnp.sqrt(alpha_) * eta
            )
            r = r + 2.0 * self.sigmaSQ * lmbda_**2 * jnp.sum(nneval**2, axis=-1) / alpha_

        log_ref_K = jax.vmap(self.ref_process.log_pdf)(y)
        log_pi_K = jax.vmap(self.Dcl.log_target)(y)
        loss = r + log_ref_K - log_pi_K

        return loss.mean()

    def train(self):
        """Train neural network component of path integral sampler model."""
        max_samples = self.config["max_samples"]  # Number of samples in the pool
        nsamples = self.config["nsamples"]  # Number of samples per mini-batch

        key = jax.random.PRNGKey(self.config["seed"])
        lr_bk = self.config["base_lr"]

        converged = False
        finalized = False
        iter = 0
        while not converged and not finalized:
            print(f"===Iter {iter+1}")
            key, subkey = jax.random.split(key)
            # Initialize samples
            # Sample from reference process
            x_pool = self.ref_process.rvs(subkey, shape=(max_samples,))
            # Train with pool batch
            nbatches = max_samples // nsamples
            for i in range(nbatches):
                print(f"=====Mini-batch {i+1}")
                x = x_pool[i * nsamples : (i + 1) * nsamples]
                train_ds = {"input": x, "label": jnp.zeros(x.shape)}
                key, subkey = jax.random.split(key)
                # Configure criterion to take current key
                self.config["criterion"] = partial(
                    self.compute_loss,
                    key=subkey,
                )
                # Train model covering all path
                key, subkey = jax.random.split(key)
                self.nnmodel, loss = train(self.config, self.nnmodel, subkey, train_ds)
            if loss < self.config["max_loss"]:
                converged = True
                finalized = True
            else:
                iter = iter + 1
                self.config["base_lr"] = self.config["base_lr"] / 2

                if iter >= self.config["max_subiter"]:
                    finalized = True

        self.config["base_lr"] = lr_bk
        save_model(self.nnmodel, self.config["root_path"], f"nnx-state-dds")
        print("===================================================")

    def sample(self, nsamples: int, subkey: ArrayLike):
        """Use trained Denoising Diffusion sampler model to sample from the target distribution.

        This involves sampling from the reference process and propagating using the trained
        network.

        Args:
            nsamples: Number of samples to generate and transport.
            subkey: JAX random generation.

        Returns:
            Samples generated from the trained model.
        """
        y = self.ref_process.rvs(subkey, shape=(nsamples,))
        ypath = [y]
        keyl = subkey

        self.nnmodel.eval()
        for k in range(self.K):
            alpha_ = self.alpha[self.K - k - 1]
            lmbda_ = 1.0 - jnp.sqrt(1.0 - alpha_)
            keyl, subkey = jax.random.split(keyl)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            dk = float(self.K - k) / self.K
            nneval = self.nnmodel(y, dk)
            # nneval = self.nnmodel(y, self.K - k)
            y = (
                (1.0 - lmbda_) * y
                + 2.0 * self.sigmaSQ * lmbda_ * nneval
                + self.sigma * jnp.sqrt(alpha_) * eta
            )
            # Store path
            ypath.append(y)

        return ypath

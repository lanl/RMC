# -*- coding: utf-8 -*-

"""Utilities for deploying a Path Integral Sampler (PIS) as
demonstrated in :cite:`zhang-2022-pis`."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.models import NN_gradient_informed, NN_with_time, NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import save_model, train


class PathIntegralSampler(nnx.Module):
    """Definition of Path Integral Sampler (PIS) class."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        h: float,
        T: int,
        verbose: bool = False,
    ):
        """Initialization of Path Integral Sampler class.

        Args:
            config: Dictionary with PIS configuration parameters.
            densitycl: Density class representing function to sample from.
            h: Time step.
            T: Number of time steps.
            verbose: Verbosity flag. Display configuration and steps if true.
        """
        super().__init__()

        # Store configuration
        self.config = config

        # Store density class representing target density function and components
        self.Dcl = densitycl

        # No need to configure sampling from initial distribution
        # Initialization is always zero

        # Store dimension
        self.d = config["dim"]
        # Store time step
        self.h = h
        self.hsqrt = jnp.sqrt(h)
        # Store time steps T
        self.T = T
        # Store terminal time
        self.TT = h * T

        # Create NN model
        if config["nn_type"] == "time_embed":
            self.nnmodel = NN_with_time_embedding(self.config)
        elif config["nn_type"] == "score":
            self.nnmodel = NN_gradient_informed(self.config, self.Dcl.der_log_target_proposal)
        else:
            self.nnmodel = NN_with_time(self.config)

    def compute_loss(self, pisnn: Callable, x: ArrayLike, y: ArrayLike, key: ArrayLike):
        """Evaluate path integral cost.

        The cost is expressed as a path cost and a terminal cost.

        Args:
            pisnn: Neural network model learned for PIS.
            x: Samples from initial distribution.
            y: Dum variable (for compatibility with trainer).
            key: JAX random generation.

        Returns:
            Current loss.
        """

        nsamples = x.shape[0]
        # Initialize y () to zero
        y = jnp.zeros(nsamples)

        for k in range(1, self.T + 1):
            t = k * self.h
            key, subkey = jax.random.split(key)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            nneval = pisnn(x, t)
            x = x + self.h * nneval + self.hsqrt * eta
            y = y + self.h * jnp.sum(nneval**2, axis=-1) / 2.0

        log_mu0_T = -jnp.sum(x**2, axis=-1) / 2.0 / self.TT
        log_mu_T = jax.vmap(self.Dcl.log_target)(x)
        y = y + log_mu0_T - log_mu_T

        return y.mean()

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
            # Train with pool batch
            nbatches = max_samples // nsamples
            for i in range(nbatches):
                print(f"=====Mini-batch {i+1}")
                # Initialize samples to zero
                x = jnp.zeros((nsamples, self.d))
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
        save_model(self.nnmodel, self.config["root_path"], f"nnx-state-pis")
        print("===================================================")

    def sample(self, nsamples: int, subkey: ArrayLike):
        """Use trained Path Integral sampler model to sample from the target distribution.

        This involves sampling from the base distribution and propagating using the trained
        network.

        Args:
            nsamples: Number of samples to generate and transport.
            subkey: JAX random generation.

        Returns:
            Samples generated from the trained model.
        """
        x = jnp.zeros((nsamples, self.d))
        # Initialize y () to zero
        y = jnp.zeros(nsamples)
        xpath = [x]
        keyl = subkey

        self.nnmodel.eval()
        for k in range(1, self.T + 1):
            t = k * self.h
            keyl, subkey = jax.random.split(keyl)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            nneval = self.nnmodel(x, t)
            # print(f"In sample --> eta.shape: {eta.shape}")
            # print(f"In sample --> nneval.shape: {nneval.shape}")
            x = x + self.h * nneval + self.hsqrt * eta
            y = y + self.h * jnp.sum(nneval**2, axis=-1) / 2.0 + self.hsqrt * eta @ nneval.T
            # Store path
            xpath.append(x)

        log_mu0_T = -jnp.sum(x**2, axis=-1) / 2.0 / self.TT
        log_mu_T = jax.vmap(self.Dcl.log_target)(x)

        y = y + log_mu0_T - log_mu_T
        w = jnp.exp(-y)
        return xpath, w

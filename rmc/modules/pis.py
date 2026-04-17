# -*- coding: utf-8 -*-

"""Utilities for deploying a Path Integral Sampler (PIS) as
demonstrated in :cite:`zhang-2022-pis`."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.models import MLP, SinusoidalPositionEmbeddings
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import save_model, train
from rmc.utils.density import LogDensityPath, LogDensityPosterior
from rmc.utils.packed_distributions import PackedMultivariateNormal


class NN_with_time_embedding(nnx.Module):
    """Definition of neural network model with time dependence via
    sinusoidal embedding."""

    def __init__(self, config: NNConfigDict, dim_sine_embedding=128):
        super().__init__()
        rngs = nnx.Rngs(config["seed"])

        dim = config["dim"]
        time_dim = dim * 4

        self.time_mlp = nnx.Sequential(
            *[
                SinusoidalPositionEmbeddings(dim_sine_embedding),
                nnx.Linear(dim_sine_embedding, time_dim, rngs=rngs),
                nnx.gelu,
                nnx.Linear(time_dim, time_dim, rngs=rngs),
            ]
        )

        self.nn = MLP(
            ndim_in=dim + time_dim,  # Additional for time dimension
            ndim_out=dim,
            layer_widths=config["layer_widths"],
            activation_func=config["activation_func"],
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike, t: float) -> ArrayLike:
        """Compute velocity field of Liouville Flow.

        Args:
            x: The position array to be evaluated.
            t: The time to be evaluated.

        Returns:
            Velocity field at current samples.
        """
        # t_ = jnp.tile(jnp.asarray(t, dtype=jnp.float32), (x.shape[0], 1))
        t_ = jnp.tile(self.time_mlp(t), (x.shape[0], 1))
        x_t = jnp.concatenate([x, t_], axis=-1)

        return self.nn(x_t)


class NN_with_time(nnx.Module):
    """Definition of neural network model with time dependence."""

    def __init__(self, config: NNConfigDict):
        super().__init__()
        rngs = nnx.Rngs(config["seed"])

        dim = config["dim"]

        self.nn = MLP(
            ndim_in=dim + 1,  # Additional for time dimension
            ndim_out=dim,
            layer_widths=config["layer_widths"],
            activation_func=config["activation_func"],
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike, t: float) -> ArrayLike:
        """Compute velocity field of Liouville Flow.

        Args:
            x: The position array to be evaluated.
            t: The time to be evaluated.

        Returns:
            Velocity field at current samples.
        """
        t_ = jnp.tile(jnp.asarray(t, dtype=jnp.float32), (x.shape[0], 1))
        x_t = jnp.concatenate([x, t_], axis=-1)

        return self.nn(x_t)


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

        # Configure sampling from initial distribution
        if isinstance(densitycl, LogDensityPath):  # Path == type 1
            # Use initial density to sample from initial distribution
            self.distribution0 = densitycl.initial.rvs
        elif isinstance(densitycl, LogDensityPosterior):  # Posterior == type 2
            # Use prior density to sample from initial distribution
            self.distribution0 = densitycl.prior.rvs
        else:
            # Use provided distribution
            # If not provided, use multivariate normal with zero mean and identity covariance
            if "dist0" in config.keys():
                self.distribution0 = config["dist0"].rvs
            else:
                d = config["dim"]
                mean_base = jnp.zeros(d).reshape((1, d))
                cov_base = jnp.eye(d).reshape((1, d, d))
                self.distribution0 = PackedMultivariateNormal(mean_base, cov_base).rvs

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
        if config["time_embed"]:
            self.nnmodel = NN_with_time_embedding(self.config)
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

        log_mu0_T = -jnp.sum(x**2, axis=-1) / 2.0 / self.T / self.h
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
            key, subkey = jax.random.split(key)
            # Initialize samples
            # Sample from initial distribution
            x_pool = self.distribution0(subkey, shape=(max_samples,))
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
        x = self.distribution0(subkey, shape=(nsamples,))
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

        log_mu0_T = -jnp.sum(x**2, axis=-1) / 2.0 / self.T / self.h
        log_mu_T = jax.vmap(self.Dcl.log_target)(x)

        y = y + log_mu0_T - log_mu_T
        w = jnp.exp(-y)
        return xpath, w

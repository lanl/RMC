# -*- coding: utf-8 -*-

"""Utilities for deploying a Denoising Diffusion Sampler (DDS) as
demonstrated in :cite:`vargas-2023-dds`."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.models import NN_gradient_informed, NN_with_time, NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import save_model, train
from rmc.utils.packed_distributions import PackedMultivariateNormal
from rmc.utils.schedule_diffusion import prepare_dds_noise_variance


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
        schedule_convention: str = "noise_variance",
        reverse_schedule: bool = True,
        control_parameterization: str = "f",
    ):
        """Initialization of Denoising Diffusion Sampler class.

        Args:
            config: Dictionary with DDS configuration parameters.
            densitycl: Density class representing function to sample from.
            sigma: Standard deviation of reference process.
            K: Number of time steps.
            beta_schedule: Function returning the discrete diffusion schedule.
                By default its outputs are interpreted directly as per-step
                noise variances.
            verbose: Verbosity flag. Display configuration and steps if true.
            schedule_convention: Interpretation of beta_schedule outputs.
                Options are "noise_variance" and "legacy_complement".
            reverse_schedule: If true, reverse the schedule into DDS
                generation order.
            control_parameterization: Neural-control parameterization.
                ``"f"`` uses the existing RMC/published DDS form, while
                ``"u"`` uses the rescaled control form employed by the
                original DDS repository.
        """
        super().__init__()

        # Store configuration
        self.config = config

        if control_parameterization not in ("f", "u"):
            raise ValueError(
                "Unsupported DDS control parameterization "
                f"{control_parameterization!r}. Expected 'f' or 'u'."
            )
        self.control_parameterization = control_parameterization

        # Store density class representing target density function and components
        self.Dcl = densitycl

        # Store dimension
        self.d = config["dim"]
        # Store time steps K
        self.K = K
        # Store raw schedule and construct DDS coefficients in generation order
        self.beta = beta_schedule(K)
        self.noise_variance = prepare_dds_noise_variance(
            self.beta,
            convention=schedule_convention,
            reverse=reverse_schedule,
        )
        self.retention = jnp.sqrt(1.0 - self.noise_variance)
        self.lmbda = 1.0 - self.retention
        # Store standard deviation and variance for reference process
        self.sigma = sigma
        self.sigmaSQ = sigma**2
        self.ref_process = PackedMultivariateNormal(
            jnp.zeros(self.d).reshape((1, self.d)),
            self.sigmaSQ * jnp.eye(self.d).reshape((1, self.d, self.d)),
        )

        # Create NN model
        if config["nn_type"] == "time_embed":
            self.nnmodel = NN_with_time_embedding(self.config)
        elif config["nn_type"] == "score":
            self.nnmodel = NN_gradient_informed(self.config, self.Dcl.der_log_target_proposal)
        else:
            self.nnmodel = NN_with_time(self.config)

    def _control_shift(
        self,
        control: ArrayLike,
        noise_variance: ArrayLike,
        lmbda: ArrayLike,
    ) -> ArrayLike:
        """Evaluate the controlled mean shift for one DDS step."""
        if self.control_parameterization == "f":
            return 2.0 * self.sigmaSQ * lmbda * control

        return noise_variance * control

    def _running_cost(
        self,
        control: ArrayLike,
        noise_variance: ArrayLike,
        lmbda: ArrayLike,
    ) -> ArrayLike:
        """Evaluate the discrete transition KL contribution."""
        control_sq = jnp.sum(control**2, axis=-1)

        if self.control_parameterization == "f":
            return (
                2.0
                * self.sigmaSQ
                * lmbda**2
                * control_sq
                / noise_variance
            )

        return noise_variance * control_sq / (2.0 * self.sigmaSQ)

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
            noise_variance = self.noise_variance[k]
            retention = self.retention[k]
            lmbda = self.lmbda[k]
            key, subkey = jax.random.split(key)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            dk = float(self.K - k) / self.K
            control = ddsnn(y, dk)
            y = (
                retention * y
                + self._control_shift(control, noise_variance, lmbda)
                + self.sigma * jnp.sqrt(noise_variance) * eta
            )
            r = r + self._running_cost(
                control,
                noise_variance,
                lmbda,
            )

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
            noise_variance = self.noise_variance[k]
            retention = self.retention[k]
            lmbda = self.lmbda[k]
            keyl, subkey = jax.random.split(keyl)
            eta = jax.random.normal(subkey, (nsamples, self.d))
            dk = float(self.K - k) / self.K
            control = self.nnmodel(y, dk)
            y = (
                retention * y
                + self._control_shift(control, noise_variance, lmbda)
                + self.sigma * jnp.sqrt(noise_variance) * eta
            )
            # Store path
            ypath.append(y)

        return ypath

# -*- coding: utf-8 -*-

"""Utilities for deploying a Conditional Flow Matching (CFM) Sampler as
demonstrated in :cite:`lipman-2023-cfm`."""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpy as np
from flax import nnx
from optax import l2_loss
from scipy.integrate import solve_ivp

from rmc.flax.models import NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import save_model, train


class ConditionalFlowMatching(nnx.Module):
    """Definition of Conditional Flow Matching (CFM) class."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        samples: ArrayLike,
        model_type: str,
        schedule: Optional[Callable] = None,
        mu_f: Optional[Callable] = None,
        sigma_f: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """Initialization of Conditional Flow Matching class.

        Args:
            config: Dictionary with CFM configuration parameters.
            densitycl: Density class representing function to sample from.
            initial_samples: Initial samples from target distribution.
            model_type: String defining the type of conditional path to use.
                Options: 've' for variance exploding, 'vp' for variance preserving
                'ot' for optimal transport. If the provided string is none or
                not recognized, then the passed sigma and mu functions are used.
                An error is generated if the model type is not recognized and mu
                and sigma functions are not passed.
            schedule: Definition of diffusion schedule. Only applies for 've'.
            mu_f: Optional function representing the mean of the probability path.
                Only used if model type is not 've', 'vp' or 'ot'.
            sigma_f: Optional function representing the standard deviation of the
                probability path. Only used if model type is not 've', 'vp' or 'ot'.
            verbose: Verbosity flag. Display configuration and steps if true.
        """
        super().__init__()

        # Store configuration
        self.config = config
        self.d = config["dim"]

        # Store density class representing target density function and components
        self.Dcl = densitycl

        # Store provided (initial) samples from target distribution
        self.initial_samples = samples

        # Build/store mu, sigma and ucond functions
        if model_type == "ve":  # variance exploding
            # Store schedule
            assert schedule is not None, "schedule not defined"
            self.schedule = schedule
            self.mu_f = lambda x, t: x
            self.sigma_f = lambda x, t: self.schedule.tau(1.0 - t)
            self.ucond_f = self.eval_ucond_ve
        elif model_type == "ot":  # optimal transport
            assert "sigma_min" in config.keys(), "sigma_min not defined"
            self.sigma_min = config["sigma_min"]
            self.mu_f = lambda x, t: t * x
            self.sigma_f = lambda x, t: 1.0 - (1.0 - self.sigma_min) * t
            self.ucond_f = self.eval_ucond_ot
        else:  # arbitrary affine transformation
            model_type = "gen"
            assert mu_f is not None, "mu_f not defined"
            assert sigma_f is not None, "sigma_f not defined"
            self.mu_f = mu_f
            self.sigma_f = sigma_f
            self.d_mu_f = jax.grad(self.mu_f, argnums=1)
            self.d_sigma_f = jax.grad(self.sigma_f, argnums=1)
            self.ucond_f = self.eval_ucond_generic

        self.model_type = model_type

        # Create base model
        self.nnmodel = NN_with_time_embedding(self.config)

    def eval_ucond_ve(self, x1: ArrayLike, x: ArrayLike, t: ArrayLike):
        """Function for variance exploding diffusion conditional vector field.

        Args:
            x1: Samples from target distribution that condition the velocity field.
            x: Spatial coordinates to evaluate the conditional velocity field.
            t: Time to evaluate the conditional velocity field.

        Returns:
            Evaluation of the conditional velocity field for variance exploding at
            the provided spatial and time coordinates.
        """
        # print(f"in eval_ucond_ve --> x1 shape: {x1.shape},  x shape: {x.shape}, t shape: {t.shape}")
        # Evaluate schedule
        tau, dtau = jax.vmap(self.schedule)(1.0 - t.squeeze())
        # print(f"in eval_ucond_ve --> tau shape: {tau.shape}, dtau shape: {dtau.shape}")
        return -dtau[:, None] / tau[:, None] * (x - x1)

    def eval_ucond_ot(self, x1: ArrayLike, x: ArrayLike, t: ArrayLike):
        """Function for optimal transport conditional vector field.

        Args:
            x1: Samples from target distribution that condition the velocity field.
            x: Spatial coordinates to evaluate the conditional velocity field.
            t: Time to evaluate the conditional velocity field.

        Returns:
            Evaluation of the conditional velocity field for variance exploding at
            the provided spatial and time coordinates.
        """
        onemsig = 1.0 - self.sigma_min
        return (x1 - onemsig * x) / (1.0 - onemsig * t)

    def eval_ucond_generic(self, x1: ArrayLike, x: ArrayLike, t: ArrayLike):
        """Function for conditional vector field for canonical transformation of
        Gaussian distributions.

        This requires definitions of sigma_f, mu_f and their derivatives.

        Args:
            x1: Samples from target distribution that condition the velocity field.
            x: Spatial coordinates to evaluate the conditional velocity field.
            t: Time to evaluate the conditional velocity field.

        Returns:
            Evaluation of the conditional velocity field for variance exploding at
            the provided spatial and time coordinates.
        """
        return (self.d_sigma_f(x1, t) / self.sigma_f(x1, t)) * (x - self.mu_f(x1, t)) + self.d_mu_f(
            x1, t
        )

    def eval_psi(self, x1: ArrayLike, x: ArrayLike, t: ArrayLike):
        """Evaluate conditioned flow.

        Args:
            x1: Samples from target distribution that condition the flow.
            x: Spatial coordinates to evaluate the conditioned flow.
            t: Time to evaluate the conditioned flow.

        Returns:
            Evaluation of the conditional flow at the provided spatial and
            time coordinates.
        """
        return self.sigma_f(x1, t) * x + self.mu_f(x1, t)

    def eval_ucond(self, x1: ArrayLike, x: ArrayLike, t: ArrayLike):
        """Evaluate conditional vector field.

        Args:
            x1: Samples from target distribution that condition the velocity field.
            x: Spatial coordinates to evaluate the conditional velocity field.
            t: Time to evaluate the conditional velocity field.

        Returns:
            Evaluation of the conditional velocity field at the provided
            spatial and time coordinates.
        """
        return self.ucond_f(x1, x, t)

    def compute_loss(self, cfmnn: Callable, x: ArrayLike, y: ArrayLike, key: ArrayLike):
        """Evaluate cost for CFM model.

        Time component is generated on the fly.

        Args:
            cfmnn: Neural network representing vector field for CFM.
            x: Samples from target distribution.
            y: Dum variable (for compatibility with trainer).
            key: JAX random generation for time.

        Returns:
            Current loss.
        """

        nsamples = x.shape[0]
        key, subkey1, subkey2 = jax.random.split(key, 3)
        # Generate random initial x0 from normal distribution
        x0 = jax.random.normal(subkey1, (nsamples, self.d))
        # Generate random uniform time in [0, 1]
        t = jax.random.uniform(subkey2, (nsamples, 1))
        # Eval conditional flow (psi) and conditional vector field (u) for x0 at time t
        psi = self.eval_psi(x, x0, t)
        ucond = self.ucond_f(x, x0, t)
        # Compute MSE
        mse = l2_loss(cfmnn(psi, t), ucond)
        return jnp.mean(mse)

    def train(self):
        """Train neural network component of conditional flow matching model."""
        max_samples = self.initial_samples.shape[0]  # Number of samples in the pool
        nsamples = min(self.config["nsamples"], max_samples)  # Number of samples per mini-batch

        key = jax.random.PRNGKey(self.config["seed"])
        lr_bk = self.config["base_lr"]

        converged = False
        finalized = False
        iter = 0
        while not converged and not finalized:
            print(f"===Iter {iter+1}")
            key, subkey = jax.random.split(key)
            perms = jax.random.permutation(subkey, max_samples)
            # Train with pool batch
            nbatches = max_samples // nsamples
            for i in range(nbatches):
                print(f"=====Mini-batch {i+1}")
                x = self.initial_samples[perms[i * nsamples : (i + 1) * nsamples]]
                train_ds = {"input": x, "label": jnp.zeros(x.shape)}
                key, subkey = jax.random.split(key)
                # Configure criterion to take current key
                self.config["criterion"] = partial(
                    self.compute_loss,
                    key=subkey,
                )
                # Train model
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
        save_model(self.nnmodel, self.config["root_path"], f"nnx-cfm-{self.model_type}")
        print("===================================================")

    def sample(self, nsamples: int, t_steps: int, subkey: ArrayLike):
        """Use trained conditional flow matching model to sample from the target distribution.

        This involves sampling from a standard Gaussian distribution and propagating using the trained
        network and a ODE solver.

        Args:
            nsamples: Number of samples to generate and transport.
            t_steps: Number of time steps in ODE solver.
            subkey: JAX random generation.

        Returns:
            Samples generated from the trained model. Returned time steps
            may be different from the number specified.
        """
        x0 = np.array(jax.random.normal(subkey, (self.d, nsamples)))

        t_span = (0.0, 1.0)
        t_eval = np.linspace(*t_span, t_steps)

        xpath = np.zeros((t_steps + 1, nsamples, self.d))
        xpath[0] = x0.transpose()

        self.nnmodel.eval()
        f_ = lambda t, x: self.nnmodel(jnp.reshape(x, (-1, self.d)), t)
        res_ = solve_ivp(
            f_, t_span, x0.flatten(), method="DOP853", t_eval=t_eval, vectorized=True, rtol=1e-4
        )
        # print(f"shape of res_ y: {res_.y.shape}")
        steps_solve = len(res_.t)
        print(f"t steps in ODE solve: {steps_solve}")
        xpath[1 : steps_solve + 1, ...] = res_.y.reshape((self.d, nsamples, steps_solve)).transpose(
            (2, 1, 0)
        )

        return xpath[: steps_solve + 1]

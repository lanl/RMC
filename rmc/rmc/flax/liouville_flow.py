# -*- coding: utf-8 -*-

"""Utilities for deploying a Liouville flow-based sampler."""

from functools import partial
from copy import deepcopy
from typing import Callable, Optional, Sequence

from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

from flax import nnx

from rmc import LogDensityPath, LogPosterior

from .nn_config_dict import NNConfigDict
from .models import MLP
from .trainer import train
from .utils import divergence


class NN_LiouvilleFlow(nnx.Module):
    """Definition of neural network model for Liouville Flow."""
    def __init__(self, config: NNConfigDict):
        super().__init__()
        self.nnlf = MLP(ndim_in=config["dim"],
                        ndim_out=config["dim"],
                        layer_widths=config["layer_widths"],
                        activation_func=config["activation_func"],
                        rngs=nnx.Rngs(config["seed"])
        )
        self.mean = jnp.zeros(config["dim"])
        
    def set_flow_mean(self, mean: ArrayLike):
        self.mean = mean
        
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Compute velocity field of Liouville Flow.
        
        Args:
            x: The array to be evaluated.
            
        Returns:
            Velocity field at current samples.
        """
        return self.nnlf(x - self.mean)
        
    def nn_divergence(self, x: ArrayLike) -> ArrayLike:
        """Compute divergence of the velocity field.
        
        Args:
            x: The array to be evaluated.
            
        Returns:
            Divergence of velocity field at current samples.
        """
        return jax.vmap(divergence(self.nnlf))(x)


class LiouvilleFlow(nnx.Module):
    """Definition of Liouville Flow (LF) class."""
    def __init__(self, config: NNConfigDict, energycl, schedule: Callable, epsilon: float = 1e-6, verbose: bool=False):
        """Initialization of Liouville Flow class.
        
        Args:
            config: Dictionary with LF configuration parameters.
            energycl: Energy class representing distribution to sample from.
            schedule: Schedule function.
            epsilon: Tolerance for (float) comparisons.
            verbose: Verbosity flag. Display configuration and steps if true.
        """
        super().__init__()
        self.LFnn = NN_LiouvilleFlow(config)
        if verbose:
            nnx.display(self.LFnn)

        # Store configuration
        self.config = config

        # Store energy class representing distributions
        self.Ecl = energycl
        
        # Define initial distribution mu0
        self.mu0 = partial(multivariate_normal,
                           mean = config["mu0_mean"],
                           cov = config["mu0_covariance"],)
        
        # Store schedule
        self.schedule = schedule
        self.der_schedule = jax.vmap(jax.grad(self.schedule))
        
        # Store tolerance
        self.epsilon = epsilon
        
        # Store lists of times and models
        self.tlst = []
        self.modellst = []
        
        
    def evaluate_score_and_rhs(self, x: ArrayLike, t: float):
        """Evaluate score function and right hand side (rhs) of velocity field
        equation.
        
        Args:
            x: Current samples.
            t: Time to evaluate.
            
        Returns:
            Arrays with evaluated score and evaluated rhs.
        """
        # Evaluate schedule
        tau, dtau = self.schedule(t)
        
        # Evaluate score
        score = self.Ecl.der_log_unposterior(x, tau)
        
        if isinstance(self.Ecl, LogDensityPath): # Transform
            log_base = self.Ecl.log_base(x)
            log_target = self.Ecl.log_target(x)
            rhs = -(dtau * log_target - dtau * log_base)
        elif isinstance(self.Ecl, LogPosterior): # Bayes
            log_target = self.Ecl.log_likelihood(x)
            rhs = -(dtau * log_target)
        
        return score, rhs
        
        
    def compute_rhs_mean(self, x: ArrayLike, t: float, w: ArrayLike):
        """Evaluate mean of right hand side (rhs) of velocity field
        equation.
        
        Args:
            x: Current samples.
            t: Time to evaluate.
            w: Importance sampler log-weights.
            
        Returns:
            Mean of RHS.
        """
        # Evaluate schedule
        tau, dtau = self.schedule(t)
        
        if isinstance(self.Ecl, LogDensityPath): # Transform
            log_base = self.Ecl.log_base(x)
            log_target = self.Ecl.log_target(x)
            rhs = -(dtau * log_target - dtau * log_base)
        elif isinstance(self.Ecl, LogPosterior): # Bayes
            log_target = self.Ecl.log_likelihood(x)
            rhs = -(dtau * log_target)
            
        #print(f"In compute_rhs_mean --> rhs.shape: {rhs.shape}")
        wexp = jnp.exp(w)
        rhs_mean = jnp.mean(rhs * wexp / wexp.mean())
        return rhs_mean


    def compute_error_loss(self, x: ArrayLike, y: ArrayLike, t: float, rhs_mean: Optional[float] = None):
        """Evaluate error or the velocity field approximation.
        
        The error is expressed as the discrepancy between left hand side (lhs) and
        right hand side (rhs) of equation.
        
        Args:
            x: Current samples.
            y: Dum variable (for compatibility with trainer).
            t: Time to evaluate.
            rhs_mean: Evaluated mean of right hand side (if available).
            
        Returns:
            Current error and
        """
        # Evaluate score and right hand side
        score, rhs = self.evaluate_score_and_rhs(x, t)
        # Evaluate divergence
        divergence = self.LFnn.nn_divergence(x)
        # Evaluate velocity
        velocity = self.LFnn(x)
        
        #print(f"shapes --> score: {score.shape}, div: {divergence.shape}, vel: {velocity.shape}")
        
        if rhs_mean is None:
            rhs_mean = rhs.mean()
        
        # Evaluate left hand side
        lhs = divergence + jnp.sum(score * velocity, axis=1)
        
        #print(f"shapes --> lhs: {lhs.shape}, rhs: {rhs.shape}, rhs_mean: {rhs_mean.shape}")
        
        error = jnp.nan_to_num(lhs - (rhs - rhs_mean), posinf = 1.0, neginf = -1.0)
        errorsq = jnp.mean(error * error)
        
        return errorsq#, errorsq / jnp.nan_to_num(rhs).var()
        
        
    def sample_noweight(self, nsamples: int, subkey: ArrayLike):
        """Resample using trained network.
        
        Args:
            nsamples: Number of samples to generate and transport.
            subkey: JAX random generation.
            
        Returns:
            Sample generated and transported.
        """
        # Sample from initial distribution mu0
        x = self.mu0(subkey, shape = (nsamples,))
        
        # Transport samples using trained velocity field models
        t = 0.0
        for i in range(len(self.tlst)):
            # Evaluate velocity field for ith step via ith NN
            velocity = self.modellst[i](x)
            # Transport sample by dt
            x = x + velocity * (self.tlst[i] - t)
            # Update t
            t = self.tlst[i]
        
        return x
      
      
    def train(self):
        """Train Liouville Flow model."""
        
        if self.config["method"] == "withoutweight":
            output = self.train_withoutweight()
            
        return output
            
            
    def train_withoutweight(self):
        """Train Liouville Flow without importance sample weights."""
        #print("Inside train_withoutweight")
        t_init = 0.                         # Start interval time
        t_end = 0.11#1.0                         # End interval time
        dt_max = self.config["dt_max"]      # Maximum time step
        nsamples = self.config["nsamples"]  # Number of samples
        
        key = jax.random.PRNGKey(self.config["seed"])
        key, subkey = jax.random.split(key)
        x = self.sample_noweight(nsamples, subkey)
        # Initialize weights to zero (no weights)
        w = jnp.zeros(nsamples)
        batch = jnp.arange(nsamples)
        
        dt = 5e-2
        t = dt#t_init
        while t < t_end - self.epsilon:
            self.LFnn.set_flow_mean(x.mean(axis=0))
            rhs_mean = self.compute_rhs_mean(x, t, w)
            print(f"Training --> t: {t}, rhs mean: {rhs_mean}")
            # Construct training set. y=0 is expected label
            train_ds = {"input": x, "label": jnp.zeros(x.shape)}
            # Configure criterion to take current time and rhs_mean
            self.config["criterion"] = partial(self.compute_error_loss, t = t,
                                        rhs_mean = rhs_mean,
                                       )
            # Train model at current step
            self.LFnn = train(self.config, self.LFnn, key, train_ds)
            # Append trained model and t
            self.tlst.append(t)
            self.modellst.append(deepcopy(self.LFnn))
            print("===================================================")
            # Increment time step
            t = t + dt


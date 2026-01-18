# -*- coding: utf-8 -*-

"""Utilities for deploying a Liouville flow-based sampler."""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from rmc.flax.models import MLP
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import load_model, save_model, train
from rmc.utils.density import LogDensityPath, LogDensityPosterior
from rmc.utils.math_utils import divergence


class NN_LiouvilleFlow(nnx.Module):
    """Definition of neural network model for Liouville Flow."""

    def __init__(self, config: NNConfigDict):
        super().__init__()
        self.nnlf = MLP(
            ndim_in=config["dim"],
            ndim_out=config["dim"],
            layer_widths=config["layer_widths"],
            activation_func=config["activation_func"],
            rngs=nnx.Rngs(config["seed"]),
        )
        self.mean = jnp.zeros(config["dim"])

    def set_flow_mean(self, mean: ArrayLike):
        """Set a current flow mean.

        This is supposed to mimic batch norm."""
        self.mean = mean

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Compute velocity field of Liouville Flow.

        Args:
            x: The array to be evaluated.

        Returns:
            Velocity field at current samples.
        """
        # return self.nnlf(x - self.mean)
        return self.nnlf(x)

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

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        schedule: Callable,
        epsilon: float = 1e-6,
        verbose: bool = False,
    ):
        """Initialization of Liouville Flow class.

        Args:
            config: Dictionary with LF configuration parameters.
            densitycl: Density class representing function to sample from.
            schedule: Schedule function.
            epsilon: Tolerance for (float) comparisons.
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
            raise NotImplementedError

        # Store schedule
        self.schedule = schedule

        # Store tolerance
        self.epsilon = epsilon

        # Store lists of times and models
        self.tlst = []
        # self.modellst = []

        # Create base model
        self.LFnn = NN_LiouvilleFlow(self.config)

    def evaluate_score(self, x: ArrayLike, t: float):
        """Evaluate score function.

        Args:
            x: Current samples.
            t: Time to evaluate.

        Returns:
            Arrays with evaluated score and evaluated rhs.
        """
        # Evaluate schedule
        tau, dtau = self.schedule(t)

        # Evaluate score
        score = self.Dcl.der_log_target_proposal(x, tau)
        return score

    def evaluate_dutlogtarget(self, x: ArrayLike, t: float):
        """Evaluate time derivative of unnormalized time-dependent log
        target (dutlogtarget) density function.

        Args:
            x: Current samples.
            t: Time to evaluate.

        Returns:
            Time derivative of nnormalized time-dependent log target
            density function evaluated at (x,t).
        """
        # Evaluate schedule
        tau, dtau = self.schedule(t)

        if isinstance(self.Dcl, LogDensityPath):  # Path == type 1
            log_initial = self.Dcl.log_initial(x)
            log_target = self.Dcl.log_target(x)
            dutlt = dtau * (log_target - log_initial)
        else:  # Density or Posterior == type 2
            log_target = self.Dcl.log_target(x)
            dutlt = dtau * log_target

        return dutlt

    def evaluate_dutlogtarget_mean(
        self, x: ArrayLike, t: float, logw: ArrayLike, dutlt: Optional[ArrayLike] = None
    ):
        """Evaluate mean of time derivative of unnormalized time-dependent log target
        (dutlogtarget) density function.

        Args:
            x: Current samples.
            t: Time to evaluate.
            logw: Log-weights of samples. (delta in paper).
            dutlt: Evaluated time derivative of unnormalized time-dependent log target density function (if available).

        Returns:
            Evaluated mean of time derivative of unnormalized time-dependent target
            density function.
        """
        if dutlt is None:
            dutlt = self.evaluate_dutlogtarget(x, t)

        # weights
        w = jnp.exp(logw)
        dutlt_mean = jnp.mean(dutlt * w / w.sum())

        return dutlt_mean

    def compute_error_loss(
        self, lfnn: Callable, x: ArrayLike, y: ArrayLike, t: float, dutlt_mean: float
    ):
        """Evaluate error or the velocity field approximation.

        The error is expressed as the discrepancy between left hand side (lhs) and
        right hand side (rhs) of equation.

        Args:
            x: Current samples.
            y: Dum variable (for compatibility with trainer).
            t: Time to evaluate.
            dutlt_mean: Mean of time derivative of unnormalized log target.

        Returns:
            Current error and error percentage(?).
        """
        # Evaluate score
        score = self.evaluate_score(x, t)
        # Evaluate time derivative of unnormalized time-dependent
        # log target (dutlogtarget) density function
        dutlt = self.evaluate_dutlogtarget(x, t)
        # Evaluate divergence
        divergence = lfnn.nn_divergence(x)
        # Evaluate velocity
        velocity = lfnn(x)

        # Evaluate left hand side of eq. 5a in paper
        lhs = divergence + jnp.sum(score * velocity, axis=1)

        # error = jnp.nan_to_num(lhs + dutlt - dutlt_mean, posinf = 1.0, neginf = -1.0) # Check this!
        error = lhs + dutlt - dutlt_mean
        errorsq = jnp.mean(error * error)

        # return errorsq, errorsq / jnp.nan_to_num(dutlt).var()
        return errorsq, errorsq / dutlt.var()

    def compute_logw_update(self, x: ArrayLike, logw: ArrayLike, t: float):
        """Compute update in log of importance sampling weight.

        Args:
            x: Current samples.
            logw: Log-weights of samples. (delta in paper).
            t: Time to evaluate.

        Returns:
            Update of log-weight, utltarget and evaluated velocity.
        """
        # Evaluate score
        score = self.evaluate_score(x, t)
        # Evaluate t derivative of unnormalized time-dependent
        # log target (dutlogtarget) density function
        dutlt = self.evaluate_dutlogtarget(x, t)
        # Evaluate divergence
        divergence = self.LFnn.nn_divergence(x)
        # Evaluate velocity
        velocity = self.LFnn(x)
        # Evaluate dutlogtarget mean
        dutlt_mean = self.evaluate_dutlogtarget_mean(x, t, logw, dutlt)
        # Evaluate log weight change
        dlw = divergence + jnp.sum(score * velocity, axis=1) + dutlt - dutlt_mean

        return dlw, dutlt_mean, velocity

    def train(self):
        """Train Liouville Flow model.

        Different weight or resampling adaptations are executed
        depending on configuration.
        """
        print(f"Training method: {self.config['method']}")
        # Default: without weighting and resampling
        weightingF = False
        resamplingF = False
        if self.config["method"] == "withweight_resample":
            weightingF = True
            resamplingF = True

        # Configure main training loop
        t_init = 0.0  # Start interval time
        t_end = 1.0  # End interval time
        dt_max = self.config["dt_max"]  # Maximum time step
        max_samples = self.config["max_samples"]  # Maximum number of samples
        nsamples = self.config["nsamples"]  # Number of samples

        key = jax.random.PRNGKey(self.config["seed"])
        key, subkey = jax.random.split(key)
        # Initialize samples
        # Sample from initial distribution
        # x_pool = self.distribution0(subkey, shape = (max_samples,))
        x_pool = self.distribution0(subkey, shape=(nsamples,))
        x = x_pool[:nsamples]
        # Initialize log weights to zero
        logw = jnp.zeros(nsamples)
        # batch = jnp.arange(nsamples)

        dt = dt_max
        t = dt  # t_init
        lr_bk = self.config["base_lr"]
        while t < t_end + self.epsilon:
            t = min(t, t_end - self.epsilon)  # Capture upper bound
            # self.LFnn.set_flow_mean(x.mean(axis=0))
            # Update t derivative of unnormalized log target
            dutlt_mean = self.evaluate_dutlogtarget_mean(x, t, logw)
            print(f"Training --> t: {t:>1.6f}, dutlt mean: {dutlt_mean:>1.2e}")

            # Construct training set.
            if resamplingF:
                key, subkey = jax.random.split(key)
                # x_pool = self.sample(max_samples, weightingF, subkey, True)
                # x = x_pool[:nsamples]
                x, logw = self.sample(nsamples, weightingF, subkey, True)
            # Label is ignored but needs to be passed for compatibility with trainer.
            train_ds = {"input": x, "label": jnp.zeros(x.shape)}
            # Configure criterion to take current time and dutlt_mean
            self.config["criterion"] = partial(
                self.compute_error_loss,
                t=t,
                dutlt_mean=dutlt_mean,
            )
            ploss = 1e4
            iter = 0
            while ploss > self.config["max_loss"] and iter < self.config["max_subiter"]:
                if iter > 0:
                    # Take different data pool
                    key, subkey = jax.random.split(key)
                    # x_pool = jax.random.permutation(subkey, x_pool)
                    # x = x_pool[:nsamples]
                    x, logw = self.sample(nsamples, weightingF, subkey, True)
                    # self.LFnn.set_flow_mean(x.mean(axis=0))
                    # dutlt_mean = self.evaluate_dutlogtarget_mean(x, t, logw)
                    # print(f"Training --> t: {t:>1.6f}, dutlt mean: {dutlt_mean:>1.2e}")
                    # self.config["criterion"] = partial(self.compute_error_loss, t = t,
                    #                            dutlt_mean = dutlt_mean,
                    #                           )
                    train_ds = {"input": x, "label": jnp.zeros(x.shape)}
                    print(f"===Iter {iter}")

                # Train model at current step
                self.LFnn, loss, ploss = train(self.config, self.LFnn, key, train_ds)
                iter = iter + 1
                self.config["base_lr"] = self.config["base_lr"] / 2

            self.config["base_lr"] = lr_bk
            # Append trained model and t
            self.tlst.append(t)
            # self.modellst.append(deepcopy(LFnn))
            # print(f"Steps: {len(self.tlst)}")
            save_model(self.LFnn, self.config["root_path"], f"nnx-state-{len(self.tlst)}")
            print("===================================================")

            # Compute updates
            dlw, dutlt_mean, velocity = self.compute_logw_update(x, logw, t)
            # Advance x
            x = x + velocity * dt
            if weightingF:
                # Advance w
                logw = logw + dlw * dt

            # Increment time step
            t = t + dt

            # Prepare velocity model for next step (if from scratch)
            if not self.config["warm_start"]:
                # Reinitialize model
                self.LFnn = NN_LiouvilleFlow(self.config)

    def sample(self, nsamples: int, withw: bool, subkey: ArrayLike, train: bool = False):
        """Use trained Liouville Flow to sample from the target distribution.

        This involves samplimg from the base distribution and propagating using trained
        networks. Importance sampling weights are used if specified.

        Args:
            nsamples: Number of samples to generate and transport.
            withw: Flag thas specifies if importance sampling weight are to be used.
            subkey: JAX random generation.
            train: Flag to specify if this is being done during training, in which case no diagnostics or path are stored.

        Returns:
            Samples generated and transported up to trained models.
        """
        # Start interval time
        t_init = 0.0

        # Sample from initial distribution mu0
        x = self.distribution0(subkey, shape=(nsamples,))
        # Initialize log weights to zero
        logw = jnp.zeros(nsamples)
        # Initialize logz
        logz = 0

        if not train:
            xpath = [x]
        # Transport samples using trained velocity field models
        tprev = 0.0
        for i, t in enumerate(self.tlst):
            # Grab model for time t
            # LFnn = self.modellst[i]
            LFnn = NN_LiouvilleFlow(self.config)
            self.LFnn = load_model(LFnn, self.config["root_path"], f"nnx-state-{i+1}")
            # LFnn.set_flow_mean(x.mean(axis=0))
            # Evaluate velocity (and components) via trained Liouville Flow
            dlw, dutlt_mean, velocity = self.compute_logw_update(x, logw, t)
            # Advance particles
            x = x + velocity * (t - tprev)
            if withw:
                # Advance weights
                logw = logw + dlw * (t - tprev)
            if not train:
                print(f"Sampling at --> t: {t:>1.6f}")
                print(f"mean velocity --> {velocity.mean():>1.2e}")
                # Advance metrics
                dlogz = -dutlt_mean
                logz = logz + dlogz * (t - tprev)
                # Store path
                xpath.append(x)
            # Advance time
            tprev = t

        # Return samples, weights in training or samples, weights and diagnostics in evaluation
        if train:
            return x, logw
        return xpath, logw, logz

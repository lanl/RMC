# -*- coding: utf-8 -*-

"""Utilities for deploying a Reverse Ornstein--Uhlenbeck (ROU) Sampler."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import optax
from flax import nnx

from rmc.flax.models import NN_with_time, NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import (
    build_optax_optimizer,
    save_model,
    train,
    train_step,
)


class ReverseOUSampler(nnx.Module):
    """Definition of Reverse Ornstein--Uhlenbeck (ROU) Sampler class."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        h: float,
        T: int,
        a_schedule: Callable,
        sigma_schedule: Callable,
        verbose: bool = False,
    ):
        """Initialization of Reverse Ornstein--Uhlenbeck Sampler class.

        The noising-time reference process is

            dY_s = -a(s) Y_s ds + sqrt(2) sigma(s) dW_s,

        with generative time t = TT - s and TT = h * T.

        The schedules are represented as piecewise constant on the numerical
        time grid, with values evaluated at interval midpoints. The Gaussian
        OU transition on each interval is then evaluated exactly.

        Args:
            config: Dictionary with neural-network configuration parameters.
            densitycl: Density class representing the target density.
            h: Time step.
            T: Number of time steps.
            a_schedule: Scalar OU drift schedule a(s).
            sigma_schedule: Scalar OU diffusion schedule sigma(s).
            verbose: Verbosity flag.
        """
        super().__init__()

        self.config = config
        self.Dcl = densitycl
        self.d = config["dim"]

        self.h = h
        self.hsqrt = jnp.sqrt(h)
        self.T = T
        self.TT = h * T

        self.a_schedule = a_schedule
        self.sigma_schedule = sigma_schedule
        self.verbose = verbose

        # Evaluate the ROU schedules at noising-time interval midpoints.
        self.s_mid = (jnp.arange(T) + 0.5) * h
        self.a_ref = jax.vmap(a_schedule)(self.s_mid)
        self.sigma_ref = jax.vmap(sigma_schedule)(self.s_mid)

        # Exact Gaussian transition coefficients for a piecewise-constant
        # OU process on each noising-time interval.
        self.ref_mean, self.ref_variance = self._build_reference_transitions()

        # Cumulative conditional moments for
        #
        #   Y_s | Y_0 = y ~ N(m(s) y, c(s) I).
        #
        self.noising_mean, self.noising_variance = self._build_conditional_moments()

        # Generative time t = TT - s.  The conditional arrays have T+1
        # entries corresponding to t_k = k h.
        self.cond_mean = self.noising_mean[::-1]
        self.cond_variance = self.noising_variance[::-1]

        # Coefficients used on each generative-time interval [t_k, t_{k+1}].
        self.a_gen = self.a_ref[::-1]
        self.sigma_gen = self.sigma_ref[::-1]

        # Create NN model for the residual drift.
        #
        # The complete learned drift will be
        #
        #   f_theta(x,t) = -sigma(TT-t)^2 x + g_theta(x,t).
        #
        # Score-informed networks are intentionally excluded because ROU
        # only requires pointwise evaluation of the target density.
        if config["nn_type"] == "time_embed":
            self.nnmodel = NN_with_time_embedding(self.config)
        elif config["nn_type"] == "score":
            raise ValueError("ROU does not use target-gradient-informed neural networks")
        else:
            self.nnmodel = NN_with_time(self.config)

    def _build_reference_transitions(self):
        """Build the Gaussian ROU transition on each noising interval."""
        a = self.a_ref
        sigma = self.sigma_ref

        mean = jnp.exp(-a * self.h)

        nonzero = jnp.abs(a) > 1.0e-12
        a_safe = jnp.where(nonzero, a, 1.0)

        variance_nonzero = sigma**2 * (-jnp.expm1(-2.0 * a * self.h)) / a_safe
        variance_zero = 2.0 * sigma**2 * self.h

        variance = jnp.where(
            nonzero,
            variance_nonzero,
            variance_zero,
        )

        return mean, variance

    def _build_conditional_moments(self):
        """Build cumulative ROU conditional mean and variance coefficients."""

        def ou_step(carry, transition):
            mean_prev, variance_prev = carry
            mean_step, variance_step = transition

            mean_next = mean_step * mean_prev
            variance_next = mean_step**2 * variance_prev + variance_step

            return (
                mean_next,
                variance_next,
            ), (
                mean_next,
                variance_next,
            )

        initial = (
            jnp.asarray(1.0),
            jnp.asarray(0.0),
        )

        _, (mean_tail, variance_tail) = jax.lax.scan(
            ou_step,
            initial,
            (self.ref_mean, self.ref_variance),
        )

        mean = jnp.concatenate(
            (
                jnp.ones((1,)),
                mean_tail,
            )
        )
        variance = jnp.concatenate(
            (
                jnp.zeros((1,)),
                variance_tail,
            )
        )

        return mean, variance

    def eval_conditional_sample(
        self,
        x_terminal: ArrayLike,
        k: ArrayLike,
        eta: ArrayLike,
    ):
        """Sample analytically from the endpoint-conditioned ROU marginal.

        At generative grid time t_k = k h,

            X_{t_k} | X_T = y
                ~ N(m_k y, c_k I).

        Args:
            x_terminal: Terminal target endpoints.
            k: Generative-time grid indices.
            eta: Standard Gaussian random variables.

        Returns:
            Samples from the ROU conditional marginal.
        """
        mean = self.cond_mean[k]
        variance = self.cond_variance[k]

        while mean.ndim < x_terminal.ndim:
            mean = mean[..., None]
            variance = variance[..., None]

        return mean * x_terminal + jnp.sqrt(variance) * eta

    def eval_conditional_drift(
        self,
        x_terminal: ArrayLike,
        x: ArrayLike,
        k: ArrayLike,
    ):
        """Evaluate the endpoint-conditioned reverse-SDE drift.

        On generative interval k, the conditional drift is

            f(x,t;y)
              = a_bar(t) x
                - 2 sigma_bar(t)^2 / c(t) * (x - m(t) y).

        The terminal grid point k=T is excluded because c(T)=0.
        """
        mean = self.cond_mean[k]
        variance = self.cond_variance[k]
        a = self.a_gen[k]
        sigma = self.sigma_gen[k]

        while mean.ndim < x.ndim:
            mean = mean[..., None]
            variance = variance[..., None]
            a = a[..., None]
            sigma = sigma[..., None]

        return a * x - 2.0 * sigma**2 / variance * (x - mean * x_terminal)

    def eval_conditional_residual(
        self,
        x: ArrayLike,
        eta: ArrayLike,
        k: ArrayLike,
    ):
        """Evaluate the regression target for the learned residual drift.

        With

            f_theta(x,t)
                = -sigma_bar(t)^2 x + g_theta(x,t),

        and the endpoint-conditioned reverse-SDE drift

            f_cond(x,t;y)
                = a_bar(t) x
                  - 2 sigma_bar(t)^2 / c(t)
                    * (x - m(t) y),

        the residual target is

            g_cond
                = [a_bar(t) + sigma_bar(t)^2] x
                  - 2 sigma_bar(t)^2 / sqrt(c(t)) eta.

        The final term is evaluated directly from eta to avoid subtractive
        cancellation in x - m(t) y near the target endpoint.
        """
        variance = self.cond_variance[k]
        a = self.a_gen[k]
        sigma = self.sigma_gen[k]

        while variance.ndim < eta.ndim:
            variance = variance[..., None]
            a = a[..., None]
            sigma = sigma[..., None]

        return (a + sigma**2) * x - 2.0 * sigma**2 / jnp.sqrt(variance) * eta

    def _log_isotropic_normal(
        self,
        x: ArrayLike,
        mean: ArrayLike,
        variance: ArrayLike,
    ):
        """Evaluate a batched isotropic Gaussian log density."""
        return -0.5 * (
            self.d * jnp.log(2.0 * jnp.pi * variance) + jnp.sum((x - mean) ** 2, axis=-1) / variance
        )

    def _network_time(self, t: ArrayLike):
        """Map physical generative time to the neural-network time input."""
        if self.config.get("rou_normalize_time", False):
            return t / self.TT

        return t

    def _eval_continuous_noising_moments(self, s: ArrayLike):
        """Evaluate forward ROU moments at arbitrary noising time.

        The configured schedules are piecewise constant on intervals of
        length ``h``.  The cumulative moments at the left grid point are
        propagated analytically through the remaining fractional interval.
        """
        s = jnp.asarray(s)
        s = jnp.clip(s, 0.0, self.TT)

        # noising_mean[j] and noising_variance[j] are the cumulative moments
        # after j complete reference intervals.
        j = jnp.floor(s / self.h).astype(jnp.int32)
        j = jnp.minimum(j, self.T - 1)

        left = j.astype(s.dtype) * self.h
        delta = s - left

        a = self.a_ref[j]
        sigma = self.sigma_ref[j]

        mean_left = self.noising_mean[j]
        variance_left = self.noising_variance[j]

        mean_step = jnp.exp(-a * delta)

        nonzero = jnp.abs(a) > 1.0e-12
        a_safe = jnp.where(nonzero, a, 1.0)

        variance_step_nonzero = sigma**2 * (-jnp.expm1(-2.0 * a * delta)) / a_safe
        variance_step_zero = 2.0 * sigma**2 * delta

        variance_step = jnp.where(
            nonzero,
            variance_step_nonzero,
            variance_step_zero,
        )

        mean = mean_step * mean_left
        variance = mean_step**2 * variance_left + variance_step

        return mean, variance, a, sigma

    def eval_conditional_sample_continuous(
        self,
        x_terminal: ArrayLike,
        s: ArrayLike,
        eta: ArrayLike,
    ):
        """Sample X_{T-s} conditional on the terminal endpoint."""
        mean, variance, _, _ = self._eval_continuous_noising_moments(s)

        while mean.ndim < x_terminal.ndim:
            mean = mean[..., None]
            variance = variance[..., None]

        return mean * x_terminal + jnp.sqrt(variance) * eta

    def eval_conditional_residual_continuous(
        self,
        x: ArrayLike,
        eta: ArrayLike,
        s: ArrayLike,
    ):
        """Evaluate the conditional residual-drift target at noising time s."""
        _, variance, a, sigma = self._eval_continuous_noising_moments(s)

        while variance.ndim < x.ndim:
            variance = variance[..., None]
            a = a[..., None]
            sigma = sigma[..., None]

        return (a + sigma**2) * x - 2.0 * sigma**2 / jnp.sqrt(variance) * eta

    def eval_drift(
        self,
        x: ArrayLike,
        k: int,
    ):
        """Evaluate the learned generative-time SDE drift.

        The drift is parameterized as

            f_theta(x,t)
                = -sigma(TT-t)^2 x + g_theta(x,t),

        where the neural network represents the residual drift g_theta.

        The baseline SDE preserves the standard normal distribution,
        independently of the time-dependent diffusion schedule.
        """
        t = k * self.h
        nn_time = self._network_time(t)

        return -self.sigma_gen[k] ** 2 * x + self.nnmodel(x, nn_time)

    def eval_proposal_transition_logpdf(
        self,
        x_prev: ArrayLike,
        x_next: ArrayLike,
        k: int,
    ):
        """Evaluate the Euler--Maruyama neural transition log density."""
        drift = self.eval_drift(x_prev, k)
        mean = x_prev + self.h * drift
        variance = 2.0 * self.sigma_gen[k] ** 2 * self.h

        return self._log_isotropic_normal(
            x_next,
            mean,
            variance,
        )

    def eval_reference_transition_logpdf(
        self,
        x_prev: ArrayLike,
        x_next: ArrayLike,
        k: int,
    ):
        """Evaluate the backward-factorized ROU transition log density.

        In generative coordinates, x_{k+1} is closer to the target than
        x_k.  Therefore r_k(x_k | x_{k+1}) is the forward noising-time
        ROU transition on interval T-k-1.
        """
        j = self.T - k - 1

        mean = self.ref_mean[j] * x_next
        variance = self.ref_variance[j]

        return self._log_isotropic_normal(
            x_prev,
            mean,
            variance,
        )

    def generate_weighted_endpoints(
        self,
        nsamples: int,
        key: ArrayLike,
    ):
        """Generate proposal paths and compute path-space importance weights.

        Proposal paths are generated from

            dX_t = f_theta(X_t,t) dt
                   + sqrt(2) sigma(TT-t) dW_t,

        with X_0 ~ N(0,I).

        For each path, the unnormalized path weight is

            W = mu_tilde(X_T)
                prod_k r_k(X_k | X_{k+1})
                /
                [phi(X_0) prod_k q_k(X_{k+1} | X_k)].

        Args:
            nsamples: Number of proposal paths.
            key: JAX random key.

        Returns:
            Terminal samples, unnormalized log weights, and diagnostics.
        """
        key, subkey = jax.random.split(key)
        x = jax.random.normal(
            subkey,
            (nsamples, self.d),
        )

        log_phi = self._log_isotropic_normal(
            x,
            jnp.zeros_like(x),
            jnp.asarray(1.0),
        )

        log_q = jnp.zeros(nsamples)
        log_r = jnp.zeros(nsamples)

        self.nnmodel.eval()

        for k in range(self.T):
            drift = self.eval_drift(x, k)

            proposal_mean = x + self.h * drift
            proposal_variance = 2.0 * self.sigma_gen[k] ** 2 * self.h

            key, subkey = jax.random.split(key)
            eta = jax.random.normal(
                subkey,
                (nsamples, self.d),
            )

            x_next = proposal_mean + jnp.sqrt(proposal_variance) * eta

            log_q = log_q + self._log_isotropic_normal(
                x_next,
                proposal_mean,
                proposal_variance,
            )

            log_r = log_r + self.eval_reference_transition_logpdf(
                x,
                x_next,
                k,
            )

            x = x_next

        log_target = jax.vmap(self.Dcl.log_target)(x)

        log_weights = log_target + log_r - log_phi - log_q

        weights = jax.nn.softmax(log_weights)
        ess = 1.0 / jnp.sum(weights**2)

        diagnostics = {
            "weights": weights,
            "ess": ess,
            "ess_fraction": ess / nsamples,
            "max_weight": jnp.max(weights),
            "log_weight_std": jnp.std(log_weights),
        }

        return x, log_weights, diagnostics

    def resample_endpoints(
        self,
        key: ArrayLike,
        x_terminal: ArrayLike,
        log_weights: ArrayLike,
        nsamples: int,
    ):
        """Resample terminal particles according to path-space weights."""
        weights = jax.nn.softmax(log_weights)

        indices = jax.random.choice(
            key,
            x_terminal.shape[0],
            shape=(nsamples,),
            replace=True,
            p=weights,
        )

        return x_terminal[indices]

    def compute_loss(
        self,
        rounn: Callable,
        x: ArrayLike,
        y: ArrayLike,
        key: ArrayLike,
    ):
        """Evaluate the analytic conditional reverse-SDE regression loss.

        The input ``x`` contains terminal samples approximating the target
        distribution.  For every endpoint, a generative-time grid point and
        Gaussian conditional perturbation are generated on the fly.

        The neural network represents the residual drift g_theta in

            f_theta(x,t)
                = -sigma(TT-t)^2 x + g_theta(x,t).

        The corresponding endpoint-conditioned regression target is

            g_cond
                = [a(TT-t) + sigma(TT-t)^2] x
                  - 2 sigma(TT-t)^2 / sqrt(c(t)) eta.

        Args:
            rounn: Neural network representing the residual ROU drift.
            x: Terminal endpoint samples.
            y: Dummy variable for compatibility with the generic trainer.
            key: JAX random key.

        Returns:
            Mean conditional regression loss.
        """
        del y

        nsamples = x.shape[0]
        key_k, key_eta = jax.random.split(key)

        # Exclude k=T because c(T)=0.
        k = jax.random.randint(
            key_k,
            shape=(nsamples,),
            minval=0,
            maxval=self.T,
        )

        eta = jax.random.normal(
            key_eta,
            shape=(nsamples, self.d),
        )

        x_cond = self.eval_conditional_sample(
            x,
            k,
            eta,
        )

        target = self.eval_conditional_residual(
            x_cond,
            eta,
            k,
        )

        # Existing RMC time-dependent networks accept an array-valued time
        # input with shape (batch, 1).
        t = (k.astype(x.dtype) * self.h)[:, None]

        prediction = rounn(
            x_cond,
            t,
        )

        return jnp.mean(0.5 * (prediction - target) ** 2)

    def _build_weighted_training_dataset(
        self,
        endpoints,
        weights,
        key,
    ):
        """Build a weighted conditional ROU regression dataset.

        Each proposal endpoint retains its normalized path-space importance
        weight. Conditional times and noises are sampled analytically from
        the ROU bridge, so no endpoint resampling is required.

        The per-example training weight is N * w_i. Since the trainer averages
        over examples, this gives the empirical objective

            sum_i w_i E_{t,eta}[loss_i].

        Multiple conditional samples per endpoint can be requested with
        ``rou_conditional_samples``.
        """
        npaths = endpoints.shape[0]
        nconditional = int(self.config.get("rou_conditional_samples", 1))

        if nconditional < 1:
            raise ValueError("rou_conditional_samples must be at least 1")

        key_time, key_eta = jax.random.split(key)

        eta = jax.random.normal(
            key_eta,
            shape=(nconditional, npaths, self.d),
            dtype=endpoints.dtype,
        )

        # Repeat endpoints across conditional observations.
        x_terminal = jnp.broadcast_to(
            endpoints[None, :, :],
            (nconditional, npaths, self.d),
        ).reshape((-1, self.d))

        eta = eta.reshape((-1, self.d))

        time_sampling = self.config.get(
            "rou_conditional_time_sampling",
            "grid",
        )

        if time_sampling == "grid":
            k = jax.random.randint(
                key_time,
                shape=(nconditional, npaths),
                minval=0,
                maxval=self.T,
            ).reshape((-1,))

            x_conditional = jax.vmap(self.eval_conditional_sample)(
                x_terminal,
                k,
                eta,
            )

            target = jax.vmap(self.eval_conditional_residual)(
                x_conditional,
                eta,
                k,
            )

            # k is the generative-time grid index.
            t = k.astype(endpoints.dtype) * self.h

        elif time_sampling == "continuous":
            epsilon = float(
                self.config.get(
                    "rou_conditional_time_epsilon",
                    1.0e-4,
                )
            )

            if not 0.0 < epsilon < self.TT:
                raise ValueError("rou_conditional_time_epsilon must lie in (0, TT)")

            # s is forward/noising time measured from the target.
            # The corresponding generative time is t = TT - s.
            s = jax.random.uniform(
                key_time,
                shape=(nconditional, npaths),
                minval=epsilon,
                maxval=self.TT,
                dtype=endpoints.dtype,
            ).reshape((-1,))

            x_conditional = self.eval_conditional_sample_continuous(
                x_terminal,
                s,
                eta,
            )

            target = self.eval_conditional_residual_continuous(
                x_conditional,
                eta,
                s,
            )

            t = self.TT - s

        else:
            raise ValueError("rou_conditional_time_sampling must be " "'grid' or 'continuous'")

        t = self._network_time(t)[:, None]

        # Normalized importance weights sum to one.  Multiplication by N keeps
        # the mean sample weight equal to one, independent of N.
        sample_weights = npaths * weights
        sample_weights = jnp.broadcast_to(
            sample_weights[None, :],
            (nconditional, npaths),
        ).reshape((-1, 1))

        # The generic trainer accepts arbitrary feature and label arrays.
        # Store time with the input and the importance weight with the label.
        train_input = jnp.concatenate(
            (x_conditional, t),
            axis=1,
        )

        train_label = jnp.concatenate(
            (target, sample_weights),
            axis=1,
        )

        return {
            "input": train_input,
            "label": train_label,
        }

    def compute_weighted_loss(
        self,
        rounn,
        x,
        y,
    ):
        """Weighted conditional ROU regression loss."""
        x_conditional = x[:, : self.d]
        t = x[:, self.d : self.d + 1]

        target = y[:, : self.d]
        sample_weights = y[:, self.d]

        prediction = rounn(x_conditional, t)

        # Match the scale of optax.l2_loss followed by a mean over dimensions.
        per_sample_loss = jnp.mean(
            0.5 * (prediction - target) ** 2,
            axis=-1,
        )

        return jnp.mean(sample_weights * per_sample_loss)

    def _train_resample_one_step(self):
        """Train using resampling and one persistent-Adam update per refresh.

        This mode reproduces the adaptive cadence used by the collaborator
        implementation:

            proposal paths
              -> path-space importance weights
              -> weighted endpoint resampling
              -> analytic conditional ROU data
              -> one optimizer update
              -> proposal refresh.

        Adam state is retained across all proposal refreshes.
        """
        npaths = int(self.config["max_samples"])
        nresample = int(
            self.config.get(
                "rou_resample_size",
                self.config.get("nsamples", npaths),
            )
        )
        nouter = int(
            self.config.get(
                "rou_outer_iterations",
                self.config.get("max_subiter", 1),
            )
        )

        if nresample < 1:
            raise ValueError("rou_resample_size must be at least 1")

        if self.config.get("has_aux", False):
            raise ValueError("resample_one_step ROU training does not support has_aux=True")

        if "lr_schedule" in self.config:
            lr_schedule_fn = self.config["lr_schedule"]
        else:
            lr_schedule_fn = optax.constant_schedule(self.config["base_lr"])

        tx = build_optax_optimizer(
            self.config,
            lr_schedule_fn,
        )
        optimizer = nnx.Optimizer(
            self.nnmodel,
            tx,
            wrt=nnx.Param,
        )

        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

        key = jax.random.PRNGKey(self.config["seed"])
        history = []

        log_every = int(
            self.config.get(
                "rou_log_every",
                self.config.get("eval_every", 1),
            )
        )
        log_every = max(log_every, 1)

        for outer in range(nouter):
            (
                key,
                path_key,
                resample_key,
                conditional_key,
            ) = jax.random.split(key, 4)

            # Generate paths from the current neural proposal.
            endpoints, log_weights, diagnostics = self.generate_weighted_endpoints(
                npaths,
                path_key,
            )

            # Match the collaborator implementation: multinomially resample
            # terminal endpoints and then treat the resampled cloud as
            # unweighted training data.
            endpoints = self.resample_endpoints(
                resample_key,
                endpoints,
                log_weights,
                nresample,
            )

            uniform_weights = jnp.full(
                (nresample,),
                1.0 / nresample,
                dtype=endpoints.dtype,
            )

            # rou_conditional_samples controls the number of independent
            # analytic bridge samples generated for every resampled endpoint.
            train_ds = self._build_weighted_training_dataset(
                endpoints,
                uniform_weights,
                conditional_key,
            )

            # Exactly one full-batch optimizer update.  Calling train_step
            # directly avoids the generic trainer epoch loop and therefore
            # also avoids the max_epochs + 1 behavior tracked in issue #8.
            self.nnmodel.train()
            loss = train_step(
                self.nnmodel,
                self.compute_weighted_loss,
                optimizer,
                metrics,
                train_ds["input"],
                train_ds["label"],
                False,
            )
            metrics.reset()

            entry = {
                "loss": float(loss),
                "ess": float(diagnostics["ess"]),
                "ess_fraction": float(diagnostics["ess_fraction"]),
                "max_weight": float(diagnostics["max_weight"]),
                "log_weight_std": float(diagnostics["log_weight_std"]),
            }
            history.append(entry)

            if outer == 0 or (outer + 1) % log_every == 0 or outer + 1 == nouter:
                print(
                    f"ROU refresh {outer + 1}/{nouter} --> "
                    f"loss: {entry['loss']:.6e}, "
                    f"ESS/N: {entry['ess_fraction']:.6f}, "
                    f"max weight: {entry['max_weight']:.6e}, "
                    f"std(log w): "
                    f"{entry['log_weight_std']:.6f}"
                )

        save_model(
            self.nnmodel,
            self.config["root_path"],
            "nnx-state-rou",
        )

        return history

    def train(self):
        """Adapt the ROU proposal.

        The default ``direct_weighted`` mode performs direct path-weighted
        regression using the generic RMC trainer.

        ``resample_one_step`` instead resamples terminal endpoints and takes
        exactly one persistent-optimizer update before refreshing the
        proposal.
        """
        training_mode = self.config.get(
            "rou_training_mode",
            "direct_weighted",
        )

        if training_mode == "resample_one_step":
            return self._train_resample_one_step()

        if training_mode != "direct_weighted":
            raise ValueError(f"Unsupported ROU training mode: {training_mode}")

        npaths = self.config["max_samples"]
        nouter = self.config.get(
            "rou_outer_iterations",
            self.config.get("max_subiter", 1),
        )

        key = jax.random.PRNGKey(self.config["seed"])
        history = []

        # Preserve a user-supplied criterion, if one exists.
        had_criterion = "criterion" in self.config
        criterion_backup = self.config.get("criterion")

        try:
            self.config["criterion"] = self.compute_weighted_loss

            for outer in range(nouter):
                print(f"===ROU outer iteration {outer + 1}/{nouter}")

                key, path_key, conditional_key, train_key = jax.random.split(
                    key,
                    4,
                )

                # Generate paths from the current neural proposal and compute
                # normalized path-space importance weights.
                endpoints, _, diagnostics = self.generate_weighted_endpoints(
                    npaths,
                    path_key,
                )

                # Direct weighted regression.  No multinomial resampling.
                train_ds = self._build_weighted_training_dataset(
                    endpoints,
                    diagnostics["weights"],
                    conditional_key,
                )

                self.nnmodel, loss = train(
                    self.config,
                    self.nnmodel,
                    train_key,
                    train_ds,
                )

                entry = {
                    "loss": float(loss),
                    "ess": float(diagnostics["ess"]),
                    "ess_fraction": float(diagnostics["ess_fraction"]),
                    "max_weight": float(diagnostics["max_weight"]),
                    "log_weight_std": float(diagnostics["log_weight_std"]),
                }

                history.append(entry)

                print(
                    "ROU proposal before update --> "
                    f"loss: {entry['loss']:.6e}, "
                    f"ESS/N: {entry['ess_fraction']:.6f}, "
                    f"max weight: {entry['max_weight']:.6e}, "
                    f"std(log w): {entry['log_weight_std']:.6f}"
                )

        finally:
            if had_criterion:
                self.config["criterion"] = criterion_backup
            else:
                self.config.pop("criterion", None)

        save_model(
            self.nnmodel,
            self.config["root_path"],
            "nnx-state-rou",
        )

        return history

    def sample(
        self,
        nsamples: int,
        subkey: ArrayLike,
    ):
        """Generate samples using the learned ROU SDE.

        Args:
            nsamples: Number of samples to generate.
            subkey: JAX random key.

        Returns:
            Complete Euler--Maruyama sample paths.
        """
        key = subkey

        key, subkey = jax.random.split(key)
        x = jax.random.normal(
            subkey,
            (nsamples, self.d),
        )

        xpath = [x]

        self.nnmodel.eval()

        for k in range(self.T):
            drift = self.eval_drift(
                x,
                k,
            )

            key, subkey = jax.random.split(key)
            eta = jax.random.normal(
                subkey,
                (nsamples, self.d),
            )

            x = x + self.h * drift + jnp.sqrt(2.0) * self.sigma_gen[k] * self.hsqrt * eta

            xpath.append(x)

        return xpath

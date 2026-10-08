# -*- coding: utf-8 -*-
# Copyright (C) 2025-2026 by RMC Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the RMC package. Details of the RMC package and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for deploying an Adjoint Schrodinger Bridge Sampler."""

import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import optax
from flax import nnx

from rmc.flax.models import NN_with_time, NN_with_time_embedding
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.trainer import build_optax_optimizer, train_step
from rmc.utils.schedule_diffusion import (
    constant_diffusion_schedule,
    constant_integrated_variance,
    geometric_diffusion_schedule,
    geometric_integrated_variance,
)


class SchrodingerBridgeSampler(nnx.Module):
    """Adjoint Schrodinger Bridge Sampler with a zero-drift reference process.

    The reference process is

        dX_t = sigma(t) dW_t,

    with accumulated variance

        Q(t) = integral_0^t sigma(s)^2 ds.

    The controller and corrector are represented by separate neural networks.
    """

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        h: float,
        T: int,
        sigma_schedule: Callable | None = None,
        integrated_variance: Callable | None = None,
        controller_seed: int | None = None,
        corrector_seed: int | None = None,
    ):
        """Initialize the Adjoint Schrodinger Bridge Sampler.

        Args:
            config: Configuration dictionary for the neural networks.
            h: Integration step size.
            T: Number of integration intervals. Paths contain ``T + 1``
                grid points and terminate at time ``h * T``. The
                ``official_repository`` implementation requires
                ``h * T == 1``.
            sigma_schedule: Optional scalar diffusion schedule.
            integrated_variance: Optional accumulated diffusion variance.
            controller_seed: Optional seed for the controller network.
            corrector_seed: Optional seed for the corrector network.

        Raises:
            ValueError: If the time discretization or diffusion schedule is invalid.
        """
        super().__init__()

        if h <= 0.0:
            raise ValueError("h must be positive")
        if T < 1:
            raise ValueError("T must be at least 1")

        self.config = dict(config)
        self.Dcl = densitycl
        self.d = config["dim"]
        self.h = float(h)
        self.T = int(T)
        self.terminal_time = self.h * self.T
        self.asbs_options = self._resolve_asbs_options()

        if self.asbs_options["implementation"] == "official_repository" and not math.isclose(
            self.terminal_time,
            1.0,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError("official_repository requires normalized time with " "h * T == 1")

        self._stage_index = 0
        self._paper_stage_index = 0

        (
            self.sigma_schedule,
            self.integrated_variance,
        ) = self._resolve_diffusion_schedule(
            sigma_schedule,
            integrated_variance,
        )

        self.base_terminal_variance = jnp.asarray(self.integrated_variance(self.terminal_time))

        if float(self.base_terminal_variance) <= 0.0:
            raise ValueError("The terminal accumulated diffusion variance must be positive.")

        controller_model_seed = self._resolve_seed(
            config["seed"],
            controller_seed,
        )
        corrector_model_seed = self._resolve_seed(
            config["seed"],
            corrector_seed,
            offset=1,
        )

        if self.asbs_options["implementation"] == "official_repository":
            self.controller = self._build_official_repository_model(controller_model_seed)
            self.corrector = self._build_official_repository_model(corrector_model_seed)
        else:
            self.controller = self._build_time_model(
                config,
                controller_model_seed,
            )
            self.corrector = self._build_static_model(
                config,
                corrector_model_seed,
            )

    def _resolve_asbs_options(self):
        """Resolve and validate ASBS algorithm options."""
        if "asbs_implementation" not in self.config:
            raise ValueError(
                "asbs_implementation must be explicitly set to " "'paper' or 'official_repository'"
            )

        implementation = self.config["asbs_implementation"]

        if implementation not in {"paper", "official_repository"}:
            raise ValueError(
                f"Unsupported ASBS implementation: {implementation!r}. "
                "Expected 'paper' or 'official_repository'."
            )

        if "asbs_control_parameterization" in self.config:
            raise ValueError(
                "asbs_control_parameterization is not configurable; "
                "it is fixed by asbs_implementation"
            )

        if implementation == "official_repository":
            replay_capacity = int(self.config.get("asbs_replay_capacity", 1000))
            replay_duplicates = int(self.config.get("asbs_replay_duplicates", 1))
            control_parameterization = "diffusion_squared"
        else:
            replay_capacity = None
            replay_duplicates = 1
            control_parameterization = "diffusion"

        if replay_capacity is not None and replay_capacity < 1:
            raise ValueError("asbs_replay_capacity must be positive")

        if replay_duplicates < 1:
            raise ValueError("asbs_replay_duplicates must be at least 1")

        init_stage = self.config.get("asbs_init_stage", "adjoint")
        if init_stage not in {"adjoint", "corrector"}:
            raise ValueError("asbs_init_stage must be 'adjoint' or 'corrector'")

        adjoint_steps = int(self.config.get("asbs_adjoint_steps", 1))
        corrector_steps = int(self.config.get("asbs_corrector_steps", 1))
        resample_size = int(self.config.get("asbs_resample_size", 512))
        resample_batch_size = int(self.config.get("asbs_resample_batch_size", 512))
        train_batch_size = int(self.config.get("asbs_train_batch_size", 512))
        train_iterations = int(self.config.get("asbs_train_iterations_per_epoch", 100))

        if adjoint_steps < 1:
            raise ValueError("asbs_adjoint_steps must be at least 1")
        if corrector_steps < 1:
            raise ValueError("asbs_corrector_steps must be at least 1")
        if resample_size < 1:
            raise ValueError("asbs_resample_size must be at least 1")
        if resample_batch_size < 1:
            raise ValueError("asbs_resample_batch_size must be at least 1")
        if train_batch_size < 1:
            raise ValueError("asbs_train_batch_size must be at least 1")
        if train_iterations < 1:
            raise ValueError("asbs_train_iterations_per_epoch must be at least 1")

        target_clip = self.config.get("asbs_target_clip", None)
        if target_clip is not None:
            target_clip = float(target_clip)
            if target_clip <= 0.0:
                raise ValueError("asbs_target_clip must be positive or None")
            if implementation != "official_repository":
                raise ValueError("asbs_target_clip is an official_repository-only option")

        model_channels = int(self.config.get("asbs_model_channels", 64))
        model_num_layers = int(self.config.get("asbs_model_num_layers", 4))

        if implementation == "official_repository":
            if model_channels < 1:
                raise ValueError("asbs_model_channels must be at least 1")
            if model_num_layers < 2:
                raise ValueError("asbs_model_num_layers must be at least 2")

        return {
            "implementation": implementation,
            "model_channels": model_channels,
            "model_num_layers": model_num_layers,
            "control_parameterization": control_parameterization,
            "diffusion_schedule": self.config.get(
                "asbs_diffusion_schedule",
                "constant",
            ),
            "sigma": self.config.get("asbs_sigma", 1.0),
            "sigma_min": self.config.get("asbs_sigma_min", 1.0e-3),
            "sigma_max": self.config.get("asbs_sigma_max", 1.0),
            "learning_rate": self.config.get("base_lr", 1.0e-3),
            "replay_capacity": replay_capacity,
            "replay_duplicates": replay_duplicates,
            "init_stage": init_stage,
            "adjoint_steps": adjoint_steps,
            "corrector_steps": corrector_steps,
            "resample_size": resample_size,
            "resample_batch_size": resample_batch_size,
            "train_batch_size": train_batch_size,
            "train_iterations": train_iterations,
            "target_clip": target_clip,
        }

    @staticmethod
    def _resolve_seed(
        default_seed: int,
        seed: int | None,
        offset: int = 0,
    ) -> int:
        """Resolve a network initialization seed."""
        if seed is not None:
            return int(seed)
        return int(default_seed) + offset

    def _build_time_model(
        self,
        config: NNConfigDict,
        seed: int,
    ):
        """Build the time-dependent controller network."""
        model_config = dict(config)
        model_config["seed"] = seed

        if config["nn_type"] == "time_embed":
            return NN_with_time_embedding(model_config)

        if config["nn_type"] == "time":
            return NN_with_time(model_config)

        raise ValueError(
            "Unsupported ASBS controller network type "
            f"{config['nn_type']!r}. Expected 'time' or 'time_embed'."
        )

    def _build_static_model(
        self,
        config: NNConfigDict,
        seed: int,
    ):
        """Build the terminal corrector network from the existing RMC model."""
        model_config = dict(config)
        model_config["seed"] = seed

        if config["nn_type"] in {"time", "time_embed"}:
            return _StaticCorrector(model_config)

        raise ValueError(
            "Unsupported ASBS corrector network type "
            f"{config['nn_type']!r}. Expected 'time' or 'time_embed'."
        )

    def _build_official_repository_model(self, seed: int):
        """Build the official low-dimensional repository Fourier MLP."""
        return _FourierMLP(
            dim=self.d,
            channels=self.asbs_options["model_channels"],
            num_layers=self.asbs_options["model_num_layers"],
            seed=seed,
        )

    def _resolve_diffusion_schedule(
        self,
        sigma_schedule,
        integrated_variance,
    ):
        """Resolve explicit or configuration-defined scalar diffusion schedules."""
        if (sigma_schedule is None) != (integrated_variance is None):
            raise ValueError(
                "sigma_schedule and integrated_variance must either both "
                "be supplied or both be omitted"
            )

        if sigma_schedule is not None:
            return sigma_schedule, integrated_variance

        schedule_name = self.asbs_options["diffusion_schedule"]

        if schedule_name == "constant":
            sigma = float(self.asbs_options["sigma"])

            if sigma <= 0.0:
                raise ValueError("asbs_sigma must be positive")

            return (
                partial(
                    constant_diffusion_schedule,
                    sigma=sigma,
                ),
                partial(
                    constant_integrated_variance,
                    sigma=sigma,
                ),
            )

        if schedule_name == "geometric":
            sigma_min = float(self.asbs_options["sigma_min"])
            sigma_max = float(self.asbs_options["sigma_max"])

            if sigma_min <= 0.0 or sigma_max <= sigma_min:
                raise ValueError(
                    "Geometric diffusion requires " "0 < asbs_sigma_min < asbs_sigma_max"
                )

            return (
                partial(
                    geometric_diffusion_schedule,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    terminal_time=self.terminal_time,
                ),
                partial(
                    geometric_integrated_variance,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    terminal_time=self.terminal_time,
                ),
            )

        raise ValueError(f"Unsupported ASBS diffusion schedule: {schedule_name}")

    def eval_sigma(self, t: ArrayLike):
        """Evaluate the scalar reference diffusion coefficient sigma(t)."""
        return self.sigma_schedule(t)

    def eval_integrated_variance(self, t: ArrayLike):
        r"""Evaluate accumulated reference variance Q(t)."""
        return self.integrated_variance(t)

    def eval_control_drift_coefficient(self, t: ArrayLike):
        r"""Evaluate the coefficient multiplying the learned controller.

        Write the controlled process as

            dX_t = [b(t, X_t) + kappa(t) u_theta(t, X_t)] dt
                   + sigma(t) dW_t.

        The ``paper`` implementation uses

            kappa(t) = sigma(t),

        while ``official_repository`` uses

            kappa(t) = sigma(t)^2.
        """
        sigma = jnp.asarray(self.eval_sigma(t))

        if self.asbs_options["control_parameterization"] == "diffusion":
            return sigma

        return sigma**2

    def eval_energy_gradient(self, x: ArrayLike):
        r"""Evaluate the terminal energy gradient.

        RMC density objects expose the score

            grad log pi(x).

        For a Boltzmann target

            pi(x) proportional to exp(-E(x)),

        the energy gradient is therefore

            grad E(x) = -grad log pi(x).

        The target density is differentiated at fixed tempering equal to
        one, matching the ordinary Boltzmann target used by ASBS.

        Args:
            x: States at which to evaluate the energy gradient.

        Returns:
            Energy gradient with the same shape as `x`.
        """
        x = jnp.asarray(x)

        return -self.Dcl.der_log_target_proposal(
            x,
            tempering=1.0,
        )

    def _clip_energy_gradient(self, energy_gradient: ArrayLike):
        r"""Apply repository-style per-sample energy-gradient clipping.

        Each terminal energy gradient is clipped to have Euclidean norm
        at most ``asbs_target_clip``. The corrector is added afterward.
        """
        energy_gradient = jnp.asarray(energy_gradient)
        target_clip = self.asbs_options["target_clip"]

        if target_clip is None:
            return energy_gradient

        norm = jax.lax.stop_gradient(
            jnp.linalg.norm(
                energy_gradient,
                axis=-1,
                keepdims=True,
            )
        )
        coefficient = jnp.minimum(
            target_clip / (norm + 1.0e-6),
            1.0,
        )

        return coefficient * energy_gradient

    def eval_terminal_adjoint(
        self,
        x_terminal: ArrayLike,
        include_corrector: bool = True,
    ):
        r"""Evaluate the terminal adjoint used by adjoint matching.

        For ASBS the terminal adjoint is

            grad E(X_T) + h(X_T),

        where ``h`` is the current terminal corrector.  The first
        adjoint-matching stage uses ``h = 0``.

        Args:
            x_terminal: Terminal states.
            include_corrector: If true, include the current corrector.
                Set to false for the first ASBS stage.

        Returns:
            Terminal adjoint with the same shape as `x_terminal`.
        """
        x_terminal = jnp.asarray(x_terminal)
        energy_gradient = self.eval_energy_gradient(x_terminal)
        adjoint = self._clip_energy_gradient(energy_gradient)

        if include_corrector:
            adjoint = adjoint + jax.lax.stop_gradient(self.eval_corrector(x_terminal))

        return adjoint

    def eval_corrector_target(
        self,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
    ):
        r"""Evaluate the paper-level corrector regression target.

        For the zero-drift reference process,

            X_t | X_0=x_0 ~ N(x_0, Q(t) I),

        so the conditional transition score at the terminal time is

            grad_{x_T} log p_base(x_T | x_0)
                = (x_0 - x_T) / Q(T).

        This is the target used by the corrector-matching step.

        Args:
            x_initial: Initial states with shape `(N, dim)`.
            x_terminal: Terminal states with shape `(N, dim)`.

        Returns:
            Conditional reference-process score with the same shape as
            `x_terminal`.

        Raises:
            ValueError: If the endpoint arrays have incompatible shapes.
        """
        x_initial = jnp.asarray(x_initial)
        x_terminal = jnp.asarray(x_terminal)

        if x_initial.shape != x_terminal.shape:
            raise ValueError("x_initial and x_terminal must have the same shape")

        return (x_initial - x_terminal) / self.base_terminal_variance

    def eval_adjoint_target(
        self,
        x_terminal: ArrayLike,
        t: ArrayLike,
        include_corrector: bool = True,
    ):
        r"""Evaluate the controller regression target for adjoint matching.

        Let the controlled drift correction be

            kappa(t) u_theta(t, x),

        where ``kappa(t)`` is fixed by the selected implementation
        profile. The network target is

            u_target(t)
                = -sigma(t)^2 / kappa(t)
                  [grad E(X_T) + h(X_T)].

        Therefore, ``paper`` uses

            u_target(t) = -sigma(t) [grad E(X_T) + h(X_T)],

        while ``official_repository`` uses

            u_target(t) = -[grad E(X_T) + h(X_T)].

        The first ASBS stage uses a zero corrector.

        Args:
            x_terminal: Terminal states.
            t: Times at which the controller target is evaluated.
            include_corrector: If true, include the current corrector.

        Returns:
            Controller regression targets with the same shape as
            ``x_terminal``.
        """
        terminal_adjoint = self.eval_terminal_adjoint(
            x_terminal,
            include_corrector=include_corrector,
        )

        if self.asbs_options["control_parameterization"] == "diffusion":
            target_scale = jnp.asarray(self.eval_sigma(t))
        else:
            target_scale = jnp.ones_like(jnp.asarray(t))

        while target_scale.ndim < terminal_adjoint.ndim:
            target_scale = target_scale[..., None]

        return -target_scale * terminal_adjoint

    def eval_bridge_moments(
        self,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
        t: ArrayLike,
    ):
        r"""Evaluate exact conditional moments of the reference process.

        For the zero-drift reference process with accumulated variance Q(t),

            X_t | (X_0=x_0, X_T=x_T)
                ~ N(m_t, V_t I),

        where

            alpha_t = Q(t) / Q(T),
            m_t = x_0 + alpha_t (x_T - x_0),
            V_t = Q(t) (1 - alpha_t).

        Args:
            x_initial: Initial states with shape `(N, dim)`.
            x_terminal: Terminal states with shape `(N, dim)`.
            t: Conditioning times with shape `(N,)` or scalar.

        Returns:
            Tuple containing the bridge mean and scalar conditional variance.

        Raises:
            ValueError: If the initial and terminal arrays have incompatible
                shapes.
        """
        x_initial = jnp.asarray(x_initial)
        x_terminal = jnp.asarray(x_terminal)
        t = jnp.asarray(t)

        if x_initial.shape != x_terminal.shape:
            raise ValueError("x_initial and x_terminal must have the same shape")

        q_t = jnp.asarray(self.integrated_variance(t))
        q_terminal = self.base_terminal_variance

        alpha = q_t / q_terminal

        while alpha.ndim < x_initial.ndim:
            alpha = alpha[..., None]

        mean = x_initial + alpha * (x_terminal - x_initial)

        variance = q_t * (1.0 - q_t / q_terminal)
        variance = jnp.maximum(variance, 0.0)

        return mean, variance

    def sample_reference_bridge(
        self,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
        t: ArrayLike,
        noise: ArrayLike,
    ):
        """Sample from the exact endpoint-conditioned reference bridge.

        Args:
            x_initial: Initial states with shape `(N, dim)`.
            x_terminal: Terminal states with shape `(N, dim)`.
            t: Conditioning times.
            noise: Standard normal noise with the same shape as the states.

        Returns:
            Samples from the conditional reference bridge.
        """
        mean, variance = self.eval_bridge_moments(
            x_initial,
            x_terminal,
            t,
        )

        noise = jnp.asarray(noise)

        return mean + jnp.sqrt(variance)[..., None] * noise

    def eval_reference_bridge_score(
        self,
        x: ArrayLike,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
        t: ArrayLike,
    ):
        r"""Evaluate the conditional reference-process score.

        For the bridge law

            p(X_t | X_0, X_T) = N(m_t, V_t I),

        the score is

            grad_x log p(X_t | X_0, X_T)
                = -(X_t - m_t) / V_t.

        At the endpoints the conditional variance vanishes, so this
        function is intended for interior bridge times.

        Args:
            x: Bridge states.
            x_initial: Initial endpoint.
            x_terminal: Terminal endpoint.
            t: Interior conditioning times.

        Returns:
            Conditional score with the same shape as `x`.
        """
        x = jnp.asarray(x)

        mean, variance = self.eval_bridge_moments(
            x_initial,
            x_terminal,
            t,
        )

        while variance.ndim < x.ndim:
            variance = variance[..., None]

        return -(x - mean) / variance

    def _format_network_time(
        self,
        t: ArrayLike,
        batch_size: int,
    ):
        """Format scalar or batched times for RMC time networks."""
        t = jnp.asarray(t, dtype=jnp.float32)

        if t.ndim == 0:
            return jnp.broadcast_to(t, (batch_size, 1))

        if t.ndim == 1:
            return t[:, None]

        return t

    def eval_control(
        self,
        x: ArrayLike,
        t: ArrayLike,
    ):
        """Evaluate the learned time-dependent controller."""
        x = jnp.asarray(x)
        nn_time = self._format_network_time(t, x.shape[0])
        return self.controller(x, nn_time)

    def eval_control_drift(
        self,
        x: ArrayLike,
        t: ArrayLike,
    ):
        r"""Evaluate the learned additive controlled drift.

        This returns

            kappa(t) u_theta(t, x),

        where ``kappa(t)`` follows the selected implementation profile.
        """
        control = self.eval_control(x, t)
        coefficient = jnp.asarray(self.eval_control_drift_coefficient(t))

        while coefficient.ndim < control.ndim:
            coefficient = coefficient[..., None]

        return coefficient * control

    def eval_corrector(self, x: ArrayLike):
        """Evaluate the learned terminal corrector."""
        x = jnp.asarray(x)

        if self.asbs_options["implementation"] == "official_repository":
            terminal_time = jnp.ones(
                (x.shape[0], 1),
                dtype=x.dtype,
            )
            return self.corrector(x, terminal_time)

        return self.corrector(x)

    def build_adjoint_batch(
        self,
        subkey: ArrayLike,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
        include_corrector: bool = True,
        terminal_adjoint: ArrayLike | None = None,
    ):
        r"""Build an adjoint-matching regression batch.

        Given endpoint pairs, this method samples a state from the exact
        reference bridge. By default, it evaluates the terminal adjoint from
        ``x_terminal``. Repository replay can instead supply a terminal
        adjoint that was frozen when the endpoint pair entered the buffer.

        Args:
            subkey: JAX random key.
            x_initial: Initial endpoint samples.
            x_terminal: Terminal endpoint samples.
            include_corrector: If true, include the current corrector when
                evaluating a terminal adjoint locally.
            terminal_adjoint: Optional precomputed terminal adjoint. When
                supplied, it is used directly and ``include_corrector`` is
                ignored.

        Returns:
            Dictionary containing controller inputs and AM targets.
        """
        x_initial = jnp.asarray(x_initial)
        x_terminal = jnp.asarray(x_terminal)

        if x_initial.shape != x_terminal.shape:
            raise ValueError("x_initial and x_terminal must have the same shape")

        if terminal_adjoint is None:
            terminal_adjoint = self.eval_terminal_adjoint(
                x_terminal,
                include_corrector=include_corrector,
            )
        else:
            terminal_adjoint = jnp.asarray(terminal_adjoint)
            if terminal_adjoint.shape != x_terminal.shape:
                raise ValueError("terminal_adjoint and x_terminal must have the same shape")
            terminal_adjoint = jax.lax.stop_gradient(terminal_adjoint)

        batch_size = x_initial.shape[0]
        _, time_key, noise_key = jax.random.split(subkey, 3)

        time_eps = jnp.finfo(x_terminal.dtype).eps * max(self.terminal_time, 1.0)

        times = jax.random.uniform(
            time_key,
            (batch_size,),
            minval=time_eps,
            maxval=self.terminal_time - time_eps,
            dtype=x_terminal.dtype,
        )

        noise = jax.random.normal(
            noise_key,
            x_terminal.shape,
            dtype=x_terminal.dtype,
        )

        bridge_samples = self.sample_reference_bridge(
            x_initial,
            x_terminal,
            times,
            noise,
        )

        if self.asbs_options["control_parameterization"] == "diffusion":
            target_scale = jnp.asarray(self.eval_sigma(times))
        else:
            target_scale = jnp.ones_like(times)

        while target_scale.ndim < terminal_adjoint.ndim:
            target_scale = target_scale[..., None]

        targets = -target_scale * terminal_adjoint

        return {
            "input": jnp.concatenate(
                [bridge_samples, times[:, None]],
                axis=-1,
            ),
            "label": targets,
        }

    def build_corrector_batch(
        self,
        x_initial: ArrayLike,
        x_terminal: ArrayLike,
    ):
        r"""Build a paper-level corrector-matching regression batch.

        The target is the terminal conditional score

            grad_{x_T} log p_base(X_T | X_0).

        Args:
            x_initial: Initial endpoint samples.
            x_terminal: Terminal endpoint samples.

        Returns:
            Dictionary containing terminal states and CM targets.
        """
        return {
            "input": jnp.asarray(x_terminal),
            "label": self.eval_corrector_target(
                x_initial,
                x_terminal,
            ),
        }

    def compute_adjoint_loss(
        self,
        model,
        input: ArrayLike,
        labels: ArrayLike,
    ):
        r"""Evaluate the paper-level adjoint-matching loss.

        The regression target is

            -sigma(t)^2 / kappa(t)
             [grad E(X_T) + h(X_T)],

        where ``kappa(t)`` is the implementation-specific controlled-drift coefficient.

        Args:
            model: Time-dependent controller model.
            input: State/time input array.
            labels: AM regression targets.

        Returns:
            Paper-mode mean half-squared Euclidean error, or
            repository-mode elementwise mean squared error.
        """
        input = jnp.asarray(input)
        labels = jnp.asarray(labels)

        x = input[:, : self.d]
        t = input[:, self.d :]

        residual = model(x, t) - labels

        if self.asbs_options["implementation"] == "official_repository":
            return jnp.mean(residual**2)

        return 0.5 * jnp.mean(jnp.sum(residual**2, axis=-1))

    def compute_corrector_loss(
        self,
        model,
        input: ArrayLike,
        labels: ArrayLike,
    ):
        r"""Evaluate the paper-level corrector-matching loss.

        Args:
            model: Terminal corrector model.
            input: Terminal states.
            labels: CM regression targets.

        Returns:
            Paper-mode mean half-squared Euclidean error, or
            repository-mode elementwise mean squared error.
        """
        input = jnp.asarray(input)
        labels = jnp.asarray(labels)

        if self.asbs_options["implementation"] == "official_repository":
            terminal_time = jnp.ones(
                (input.shape[0], 1),
                dtype=input.dtype,
            )
            prediction = model(input, terminal_time)
            residual = prediction - labels
            return jnp.mean(residual**2)

        residual = model(input) - labels

        return 0.5 * jnp.mean(jnp.sum(residual**2, axis=-1))

    def _build_stage_optimizer(self, model):
        """Build an optimizer for a paper-level matching stage."""
        if "lr_schedule" in self.config:
            learning_rate_fn = self.config["lr_schedule"]
        else:
            learning_rate_fn = optax.constant_schedule(self.asbs_options["learning_rate"])

        tx = build_optax_optimizer(
            self.config,
            learning_rate_fn,
        )

        return nnx.Optimizer(
            model,
            tx,
            wrt=nnx.Param,
        )

    def _integration_time_grid(self):
        r"""Return the Euler integration grid.

        Both implementations follow the RMC convention: ``h`` is the
        integration step size and ``T`` is the number of integration
        intervals. The grid therefore contains ``T + 1`` points on
        ``[0, h * T]``.

        The ``official_repository`` profile additionally requires
        ``h * T == 1``.
        """
        return jnp.linspace(
            0.0,
            self.terminal_time,
            self.T + 1,
        )

    def generate_paths(
        self,
        nsamples: int,
        subkey: ArrayLike,
        x_initial: ArrayLike | None = None,
    ):
        r"""Generate controlled Euler-Maruyama sample paths.

        The controlled process is

            dX_t = kappa(t) u_theta(X_t,t) dt + sigma(t) dW_t,

        where ``kappa(t)`` is either ``sigma(t)`` or ``sigma(t)^2``
        according to the selected implementation profile.

        Args:
            nsamples: Number of trajectories.
            subkey: JAX random key.
            x_initial: Optional initial states. Defaults to zero.

        Returns:
            List of states on the integration grid.
        """
        key = subkey

        if x_initial is None:
            x = jnp.zeros(
                (nsamples, self.d),
                dtype=jnp.float32,
            )
        else:
            x = jnp.asarray(x_initial)

            if x.shape != (nsamples, self.d):
                raise ValueError("x_initial must have shape (nsamples, dim)")

        xpath = [x]

        self.controller.eval()

        times = self._integration_time_grid()

        for t, t_next in zip(times[:-1], times[1:]):
            dt = t_next - t
            control_drift = self.eval_control_drift(x, t)
            sigma = jnp.asarray(self.eval_sigma(t))

            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(
                noise_key,
                x.shape,
                dtype=x.dtype,
            )

            x = x + dt * control_drift + sigma * jnp.sqrt(dt) * noise
            xpath.append(x)

        return xpath

    def generate_endpoints(
        self,
        nsamples: int,
        subkey: ArrayLike,
        x_initial: ArrayLike | None = None,
    ):
        """Generate terminal states from the controlled reference process."""
        return self.generate_paths(
            nsamples,
            subkey,
            x_initial=x_initial,
        )[-1]

    def _train_official_repository_corrector_stage(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        inner_steps: int = 1,
        batch_size: int | None = None,
        fresh_samples: int | None = None,
        use_reference_process: bool = False,
        source_sampler: Callable | None = None,
    ):
        r"""Run one replay-backed repository-style corrector stage.

        Each repository epoch obtains fresh source states, generates fresh
        endpoint pairs, appends them to persistent replay, rebuilds the
        retained replay dataset, and performs the configured number of
        optimizer updates.

        Args:
            x_initial: Empirical source pool. It is used when
                ``source_sampler`` is not supplied.
            subkey: JAX random key.
            inner_steps: Number of repository-style corrector epochs.
            batch_size: Number of replay samples per optimizer update.
                Defaults to ``asbs_train_batch_size``.
            fresh_samples: Number of fresh endpoint pairs added per epoch.
                Defaults to the size of ``x_initial``.
            use_reference_process: If true, generate endpoints using the
                uncontrolled reference process. This is needed throughout
                the initial corrector stage in the official implementation.
            source_sampler: Optional callable with signature
                ``source_sampler(key, nsamples)``. When supplied, it draws
                fresh source states during every repository epoch.

        Returns:
            Dictionary containing corrector loss history and replay size.
        """
        if self.asbs_options["implementation"] != "official_repository":
            raise ValueError(
                "_train_official_repository_corrector_stage requires "
                "asbs_implementation='official_repository'"
            )

        if inner_steps < 1:
            raise ValueError("inner_steps must be at least 1")

        x_initial = jnp.asarray(x_initial)

        if x_initial.ndim != 2 or x_initial.shape[1] != self.d:
            raise ValueError("x_initial must have shape (N, dim)")

        n_fresh = x_initial.shape[0]
        if fresh_samples is not None:
            n_fresh = int(fresh_samples)

        if n_fresh < 1:
            raise ValueError("fresh_samples must be at least 1")

        if batch_size is None:
            train_batch_size = self.asbs_options["train_batch_size"]
        else:
            train_batch_size = int(batch_size)

        if train_batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        _, optimizer = self._get_stage_optimizers()

        def build_batch(batch, key):
            del key
            return self.build_corrector_batch(
                batch["x_initial"],
                batch["x_terminal"],
            )

        key = subkey
        losses = []

        for _ in range(inner_steps):
            key, source_key, path_key, batch_key = jax.random.split(key, 4)

            source = self._sample_source(
                x_initial,
                source_key,
                n_fresh,
                source_sampler=source_sampler,
            )

            x_terminal = self._generate_training_endpoints(
                source,
                path_key,
                resample_size=n_fresh,
                resample_batch_size=self.asbs_options["resample_batch_size"],
                reference=use_reference_process,
            )

            self._corrector_replay.add(
                {
                    "x_initial": source,
                    "x_terminal": x_terminal,
                }
            )

            dataset = self._corrector_replay.build_dataset(
                duplicates=self.asbs_options["replay_duplicates"]
            )

            epoch_losses, _ = self._train_epoch(
                self.corrector,
                self.compute_corrector_loss,
                optimizer,
                dataset,
                batch_key,
                build_batch,
                train_batch_size,
            )
            losses.append(jnp.mean(epoch_losses))

        return {
            "corrector_loss": jnp.asarray(losses),
            "buffer_size": len(self._corrector_replay),
        }

    def generate_reference_paths(
        self,
        nsamples: int,
        subkey: ArrayLike,
        x_initial: ArrayLike | None = None,
    ):
        r"""Generate paths from the uncontrolled zero-drift reference SDE."""
        key = subkey

        if x_initial is None:
            x = jnp.zeros(
                (nsamples, self.d),
                dtype=jnp.float32,
            )
        else:
            x = jnp.asarray(x_initial)

            if x.shape != (nsamples, self.d):
                raise ValueError("x_initial must have shape (nsamples, dim)")

        xpath = [x]

        times = self._integration_time_grid()

        for t, t_next in zip(times[:-1], times[1:]):
            dt = t_next - t
            sigma = jnp.asarray(self.eval_sigma(t))

            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(
                noise_key,
                x.shape,
                dtype=x.dtype,
            )

            x = x + sigma * jnp.sqrt(dt) * noise
            xpath.append(x)

        return xpath

    def generate_reference_endpoints(
        self,
        nsamples: int,
        subkey: ArrayLike,
        x_initial: ArrayLike | None = None,
    ):
        """Generate terminal states from the uncontrolled reference SDE."""
        return self.generate_reference_paths(
            nsamples,
            subkey,
            x_initial=x_initial,
        )[-1]

    def _initialize_training_state(self):
        """Initialize persistent replay buffers and optimizers."""
        if not hasattr(self, "_adjoint_replay"):
            self._adjoint_replay = _ASBSReplayBuffer(capacity=self.asbs_options["replay_capacity"])

        if not hasattr(self, "_corrector_replay"):
            self._corrector_replay = _ASBSReplayBuffer(
                capacity=self.asbs_options["replay_capacity"]
            )

        if not hasattr(self, "_controller_optimizer"):
            self._controller_optimizer = self._build_stage_optimizer(self.controller)

        if not hasattr(self, "_corrector_optimizer"):
            self._corrector_optimizer = self._build_stage_optimizer(self.corrector)

    def _get_stage_optimizers(self):
        """Return persistent repository-style optimizers."""
        self._initialize_training_state()
        return (
            self._controller_optimizer,
            self._corrector_optimizer,
        )

    def _generate_training_endpoints(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        resample_size: int,
        resample_batch_size: int,
        reference: bool = False,
    ):
        """Generate fresh repository-stage endpoints in chunks."""
        x_initial = jnp.asarray(x_initial)

        if x_initial.ndim != 2 or x_initial.shape[1] != self.d:
            raise ValueError("x_initial must have shape (N, dim)")

        if resample_size < 1:
            raise ValueError("resample_size must be at least 1")

        if resample_batch_size < 1:
            raise ValueError("resample_batch_size must be at least 1")

        if resample_size <= x_initial.shape[0]:
            source = x_initial[:resample_size]
        else:
            repeats = (resample_size + x_initial.shape[0] - 1) // x_initial.shape[0]
            source = jnp.tile(
                x_initial,
                (repeats, 1),
            )[:resample_size]

        nchunks = (resample_size + resample_batch_size - 1) // resample_batch_size
        keys = jax.random.split(subkey, nchunks)

        endpoints = []

        for index, key in enumerate(keys):
            start = index * resample_batch_size
            stop = min(
                start + resample_batch_size,
                resample_size,
            )
            source_chunk = source[start:stop]

            if reference:
                endpoint_chunk = self.generate_reference_endpoints(
                    source_chunk.shape[0],
                    key,
                    x_initial=source_chunk,
                )
            else:
                endpoint_chunk = self.generate_endpoints(
                    source_chunk.shape[0],
                    key,
                    x_initial=source_chunk,
                )

            endpoints.append(endpoint_chunk)

        return jnp.concatenate(endpoints, axis=0)

    def _sample_source(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        nsamples: int,
        source_sampler: Callable | None = None,
    ):
        """Obtain one fresh repository-epoch source batch.

        If no sampler is supplied, ``x_initial`` is treated as an empirical
        source pool and is sliced or tiled deterministically.
        """
        x_initial = jnp.asarray(x_initial)

        if nsamples < 1:
            raise ValueError("nsamples must be at least 1")

        if source_sampler is not None:
            source = jnp.asarray(source_sampler(subkey, nsamples))
        elif nsamples <= x_initial.shape[0]:
            source = x_initial[:nsamples]
        else:
            repeats = (nsamples + x_initial.shape[0] - 1) // x_initial.shape[0]
            source = jnp.tile(
                x_initial,
                (repeats, 1),
            )[:nsamples]

        if source.shape != (nsamples, self.d):
            raise ValueError("source_sampler must return shape (nsamples, dim)")

        return source

    def _build_minibatch_schedule(
        self,
        n_data: int,
        batch_key: ArrayLike,
        train_batch_size: int,
        train_iterations: int,
    ):
        r"""Build a repository-style shuffled minibatch schedule.

        Each traversal starts from a fresh random permutation and consumes
        every dataset index without replacement. The final batch in a
        traversal may be smaller than ``train_batch_size``. If more optimizer
        updates are required, a new permutation starts the next traversal.

        Args:
            n_data: Number of examples in the expanded replay dataset.
            batch_key: JAX random key.
            train_batch_size: Maximum number of examples per update.
            train_iterations: Number of optimizer updates to schedule.

        Returns:
            A list of ``(indices, builder_key)`` pairs and the updated random
            key.
        """
        n_data = int(n_data)
        train_batch_size = int(train_batch_size)
        train_iterations = int(train_iterations)

        if n_data < 1:
            raise ValueError("n_data must be at least 1")

        if train_batch_size < 1:
            raise ValueError("train_batch_size must be at least 1")

        if train_iterations < 1:
            raise ValueError("train_iterations must be at least 1")

        schedule = []
        key = batch_key

        while len(schedule) < train_iterations:
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(
                permutation_key,
                n_data,
            )

            for start in range(0, n_data, train_batch_size):
                if len(schedule) >= train_iterations:
                    break

                stop = min(
                    start + train_batch_size,
                    n_data,
                )
                indices = permutation[start:stop]

                key, builder_key = jax.random.split(key)
                schedule.append((indices, builder_key))

        return schedule, key

    def _train_epoch(
        self,
        model,
        loss_fn,
        optimizer,
        dataset,
        batch_key,
        builder,
        train_batch_size: int,
    ):
        """Run one repository-style shuffled matcher epoch."""
        n_data = len(dataset["x_initial"])
        train_iterations = self.asbs_options["train_iterations"]

        if n_data < 1:
            raise ValueError("Cannot train on an empty replay dataset")

        schedule, batch_key = self._build_minibatch_schedule(
            n_data=n_data,
            batch_key=batch_key,
            train_batch_size=train_batch_size,
            train_iterations=train_iterations,
        )

        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )
        losses = []

        for indices, builder_key in schedule:
            batch = {key: value[indices] for key, value in dataset.items()}
            train_ds = builder(batch, builder_key)

            model.train()

            loss = train_step(
                model,
                loss_fn,
                optimizer,
                metrics,
                train_ds["input"],
                train_ds["label"],
                False,
            )

            metrics.reset()
            losses.append(loss)

        return jnp.asarray(losses), batch_key

    def _train_official_repository_adjoint_stage(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        inner_steps: int = 1,
        batch_size: int | None = None,
        fresh_samples: int | None = None,
        source_sampler: Callable | None = None,
        is_initial_stage: bool = False,
    ):
        r"""Run one replay-backed repository-style adjoint stage.

        Each epoch obtains fresh source states, generates endpoint pairs,
        freezes their terminal adjoints, appends all data to replay, rebuilds
        the replay dataset, and performs the configured optimizer updates.

        Args:
            x_initial: Empirical source pool used when ``source_sampler`` is
                not supplied.
            subkey: JAX random key.
            inner_steps: Number of repository-style adjoint epochs.
            batch_size: Number of replay samples per optimizer update.
                Defaults to ``asbs_train_batch_size``.
            fresh_samples: Number of fresh endpoint pairs added per epoch.
                Defaults to the size of ``x_initial``.
            source_sampler: Optional callable with signature
                ``source_sampler(key, nsamples)``.
            is_initial_stage: If true, omit the corrector from every newly
                generated terminal adjoint in this initial adjoint block.

        Returns:
            Dictionary containing adjoint loss history and replay size.
        """
        if self.asbs_options["implementation"] != "official_repository":
            raise ValueError(
                "_train_official_repository_adjoint_stage requires "
                "asbs_implementation='official_repository'"
            )

        if inner_steps < 1:
            raise ValueError("inner_steps must be at least 1")

        x_initial = jnp.asarray(x_initial)

        if x_initial.ndim != 2 or x_initial.shape[1] != self.d:
            raise ValueError("x_initial must have shape (N, dim)")

        n_fresh = x_initial.shape[0]
        if fresh_samples is not None:
            n_fresh = int(fresh_samples)

        if n_fresh < 1:
            raise ValueError("fresh_samples must be at least 1")

        if batch_size is None:
            train_batch_size = self.asbs_options["train_batch_size"]
        else:
            train_batch_size = int(batch_size)

        if train_batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        optimizer, _ = self._get_stage_optimizers()

        def build_batch(batch, key):
            return self.build_adjoint_batch(
                key,
                batch["x_initial"],
                batch["x_terminal"],
                terminal_adjoint=batch["terminal_adjoint"],
            )

        key = subkey
        losses = []

        for _ in range(inner_steps):
            key, source_key, path_key, batch_key = jax.random.split(
                key,
                4,
            )

            source = self._sample_source(
                x_initial,
                source_key,
                n_fresh,
                source_sampler=source_sampler,
            )

            x_terminal = self._generate_training_endpoints(
                source,
                path_key,
                resample_size=n_fresh,
                resample_batch_size=self.asbs_options["resample_batch_size"],
            )

            terminal_adjoint = jax.lax.stop_gradient(
                self.eval_terminal_adjoint(
                    x_terminal,
                    include_corrector=not is_initial_stage,
                )
            )

            self._adjoint_replay.add(
                {
                    "x_initial": source,
                    "x_terminal": x_terminal,
                    "terminal_adjoint": terminal_adjoint,
                }
            )

            dataset = self._adjoint_replay.build_dataset(
                duplicates=self.asbs_options["replay_duplicates"]
            )

            epoch_losses, _ = self._train_epoch(
                self.controller,
                self.compute_adjoint_loss,
                optimizer,
                dataset,
                batch_key,
                build_batch,
                train_batch_size,
            )
            losses.append(jnp.mean(epoch_losses))

        return {
            "adjoint_loss": jnp.asarray(losses),
            "buffer_size": len(self._adjoint_replay),
        }

    def train_stages(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        outer_stages: int = 1,
        adjoint_steps: int | None = None,
        corrector_steps: int | None = None,
        batch_size: int | None = None,
        fresh_samples: int | None = None,
        init_corrector_from_reference: bool | None = None,
        source_sampler: Callable | None = None,
    ):
        r"""Train ASBS using the selected implementation.

        ``paper`` interprets each outer stage as one complete adjoint-and-
        corrector cycle. ``official_repository`` interprets each outer stage
        as one matcher block and alternates between adjoint and corrector
        blocks while preserving replay and optimizer state.

        Args:
            x_initial: Initial source samples with shape ``(N, dim)``.
            subkey: JAX random key.
            outer_stages: Number of implementation-specific outer stages.
            adjoint_steps: Adjoint updates or epochs per applicable stage.
                If None, use the configured value.
            corrector_steps: Corrector updates or epochs per applicable stage.
                If None, use the configured value.
            batch_size: Official-repository replay batch size. It is not
                supported by the paper implementation.
            fresh_samples: Number of fresh official-repository trajectories
                generated per epoch. It is not supported by the paper
                implementation.
            init_corrector_from_reference: Whether an initial official-
                repository corrector block uses the reference process. If
                None, the official-repository default is true. It is not
                supported by the paper implementation.
            source_sampler: Optional official-repository source sampler with
                signature ``source_sampler(key, nsamples)``.

        Returns:
            Dictionary containing stage labels and adjoint/corrector losses.
            Official-repository histories additionally contain replay sizes.
        """
        if outer_stages < 1:
            raise ValueError("outer_stages must be at least 1")

        implementation = self.asbs_options["implementation"]

        if implementation == "official_repository":
            initialize_from_reference = True
            if init_corrector_from_reference is not None:
                initialize_from_reference = bool(init_corrector_from_reference)

            return self._train_official_repository_stages(
                x_initial,
                subkey,
                outer_stages=outer_stages,
                adjoint_steps=adjoint_steps,
                corrector_steps=corrector_steps,
                batch_size=batch_size,
                fresh_samples=fresh_samples,
                init_corrector_from_reference=initialize_from_reference,
                source_sampler=source_sampler,
            )

        if batch_size is not None:
            raise ValueError(
                "batch_size is only supported by " "asbs_implementation='official_repository'"
            )
        if fresh_samples is not None:
            raise ValueError(
                "fresh_samples is only supported by " "asbs_implementation='official_repository'"
            )
        if init_corrector_from_reference is not None:
            raise ValueError(
                "init_corrector_from_reference is only supported by "
                "asbs_implementation='official_repository'"
            )
        if source_sampler is not None:
            raise ValueError(
                "source_sampler is only supported by " "asbs_implementation='official_repository'"
            )

        if adjoint_steps is None:
            adjoint_steps = self.asbs_options["adjoint_steps"]
        if corrector_steps is None:
            corrector_steps = self.asbs_options["corrector_steps"]

        if adjoint_steps < 1 or corrector_steps < 1:
            raise ValueError("adjoint_steps and corrector_steps must be at least 1")

        key = subkey
        adjoint_losses = []
        corrector_losses = []

        for _ in range(outer_stages):
            key, stage_key = jax.random.split(key)
            result = self._train_paper_stage(
                x_initial,
                stage_key,
                adjoint_steps=adjoint_steps,
                corrector_steps=corrector_steps,
            )
            adjoint_losses.append(result["adjoint_loss"])
            corrector_losses.append(result["corrector_loss"])

        return {
            "stage": ["adjoint_corrector"] * outer_stages,
            "adjoint_loss": adjoint_losses,
            "corrector_loss": corrector_losses,
        }

    def _train_official_repository_stages(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        outer_stages: int = 2,
        adjoint_steps: int | None = None,
        corrector_steps: int | None = None,
        batch_size: int | None = None,
        fresh_samples: int | None = None,
        init_corrector_from_reference: bool = True,
        source_sampler: Callable | None = None,
    ):
        r"""Run the persistent repository-style alternating ASBS loop.

        Stages alternate between adjoint and corrector matching. Replay,
        optimizer state, and stage progression persist across calls.

        Args:
            x_initial: Initial source samples.
            subkey: JAX random key.
            outer_stages: Number of alternating AM/CM stages.
            adjoint_steps: AM epochs per AM stage. If None, use the
                configured repository value.
            corrector_steps: CM epochs per CM stage. If None, use the
                configured repository value.
            batch_size: Replay batch size.
            fresh_samples: Number of fresh trajectories added per epoch.
            init_corrector_from_reference: Use the uncontrolled reference
                process throughout an initial corrector block.
            source_sampler: Optional callable with signature
                ``source_sampler(key, nsamples)``.

        Returns:
            Stage labels, losses, and replay-buffer sizes.
        """
        if self.asbs_options["implementation"] != "official_repository":
            raise ValueError(
                "_train_official_repository_stages requires asbs_implementation='official_repository'"
            )

        if outer_stages < 1:
            raise ValueError("outer_stages must be at least 1")

        if adjoint_steps is None:
            adjoint_steps = self.asbs_options["adjoint_steps"]

        if corrector_steps is None:
            corrector_steps = self.asbs_options["corrector_steps"]

        if adjoint_steps < 1 or corrector_steps < 1:
            raise ValueError("adjoint_steps and corrector_steps must be at least 1")

        x_initial = jnp.asarray(x_initial)

        if x_initial.ndim != 2 or x_initial.shape[1] != self.d:
            raise ValueError("x_initial must have shape (N, dim)")

        self._initialize_training_state()

        key = subkey
        stage_names = []
        adjoint_losses = []
        corrector_losses = []
        adjoint_buffer_sizes = []
        corrector_buffer_sizes = []

        start_with_adjoint = self.asbs_options["init_stage"] == "adjoint"

        for _ in range(outer_stages):
            key, stage_key = jax.random.split(key)

            stage_index = self._stage_index
            is_initial_stage = stage_index == 0

            if start_with_adjoint:
                is_adjoint_stage = stage_index % 2 == 0
            else:
                is_adjoint_stage = stage_index % 2 == 1

            if is_adjoint_stage:
                result = self._train_official_repository_adjoint_stage(
                    x_initial,
                    stage_key,
                    inner_steps=adjoint_steps,
                    batch_size=batch_size,
                    fresh_samples=fresh_samples,
                    source_sampler=source_sampler,
                    is_initial_stage=is_initial_stage,
                )

                stage_names.append("adjoint")
                adjoint_losses.append(result["adjoint_loss"])
                corrector_losses.append(jnp.asarray([]))
            else:
                result = self._train_official_repository_corrector_stage(
                    x_initial,
                    stage_key,
                    inner_steps=corrector_steps,
                    batch_size=batch_size,
                    fresh_samples=fresh_samples,
                    use_reference_process=(
                        init_corrector_from_reference
                        and is_initial_stage
                        and not start_with_adjoint
                    ),
                    source_sampler=source_sampler,
                )

                stage_names.append("corrector")
                adjoint_losses.append(jnp.asarray([]))
                corrector_losses.append(result["corrector_loss"])

            adjoint_buffer_sizes.append(len(self._adjoint_replay))
            corrector_buffer_sizes.append(len(self._corrector_replay))
            self._stage_index += 1

        return {
            "stage": stage_names,
            "adjoint_loss": adjoint_losses,
            "corrector_loss": corrector_losses,
            "adjoint_buffer_size": adjoint_buffer_sizes,
            "corrector_buffer_size": corrector_buffer_sizes,
        }

    def _train_paper_stage(
        self,
        x_initial: ArrayLike,
        subkey: ArrayLike,
        adjoint_steps: int = 1,
        corrector_steps: int = 1,
    ):
        r"""Perform one paper-level ASBS alternating stage.

        The first call explicitly omits the corrector from every AM target.
        Later calls include the currently trained corrector.

        Args:
            x_initial: Initial source samples.
            subkey: JAX random key.
            adjoint_steps: Number of AM optimization steps.
            corrector_steps: Number of CM optimization steps.

        Returns:
            Dictionary containing AM and CM loss histories.
        """
        if adjoint_steps < 1 or corrector_steps < 1:
            raise ValueError("adjoint_steps and corrector_steps must be at least 1")

        x_initial = jnp.asarray(x_initial)

        if x_initial.ndim != 2 or x_initial.shape[1] != self.d:
            raise ValueError("x_initial must have shape (N, dim)")

        is_initial_stage = self._paper_stage_index == 0

        path_key, am_key, cm_path_key, cm_key = jax.random.split(
            subkey,
            4,
        )

        controller_optimizer = self._build_stage_optimizer(self.controller)
        corrector_optimizer = self._build_stage_optimizer(self.corrector)

        controller_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )
        corrector_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

        x_terminal = self.generate_endpoints(
            x_initial.shape[0],
            path_key,
            x_initial=x_initial,
        )

        adjoint_losses = []

        for _ in range(adjoint_steps):
            am_key, batch_key = jax.random.split(am_key)

            train_ds = self.build_adjoint_batch(
                batch_key,
                x_initial,
                x_terminal,
                include_corrector=not is_initial_stage,
            )

            self.controller.train()

            loss = train_step(
                self.controller,
                self.compute_adjoint_loss,
                controller_optimizer,
                controller_metrics,
                train_ds["input"],
                train_ds["label"],
                False,
            )
            controller_metrics.reset()
            adjoint_losses.append(loss)

        x_terminal = self.generate_endpoints(
            x_initial.shape[0],
            cm_path_key,
            x_initial=x_initial,
        )

        corrector_losses = []

        for _ in range(corrector_steps):
            cm_key, batch_key = jax.random.split(cm_key)

            train_ds = self.build_corrector_batch(
                x_initial,
                x_terminal,
            )

            self.corrector.train()

            loss = train_step(
                self.corrector,
                self.compute_corrector_loss,
                corrector_optimizer,
                corrector_metrics,
                train_ds["input"],
                train_ds["label"],
                False,
            )
            corrector_metrics.reset()
            corrector_losses.append(loss)

        self._paper_stage_index += 1

        return {
            "adjoint_loss": jnp.asarray(adjoint_losses),
            "corrector_loss": jnp.asarray(corrector_losses),
        }


class _ASBSReplayBuffer:
    """FIFO replay buffer for ASBS endpoint training data."""

    def __init__(self, capacity: int | None = None):
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive or None")

        self.capacity = capacity
        self._batches = []

    def add(self, batch):
        """Add one endpoint batch to the replay buffer."""
        if not batch:
            raise ValueError("batch must not be empty")

        keys = tuple(batch.keys())

        if not self._batches:
            self._keys = keys
        elif keys != self._keys:
            raise ValueError("All replay batches must have the same keys")

        self._batches.append({key: jnp.asarray(value) for key, value in batch.items()})

        if self.capacity is not None:
            combined = {
                key: jnp.concatenate(
                    [item[key] for item in self._batches],
                    axis=0,
                )
                for key in self._keys
            }

            n_keep = min(self.capacity, len(next(iter(combined.values()))))

            self._batches = [{key: value[-n_keep:] for key, value in combined.items()}]

    def build_dataset(self, duplicates: int = 1):
        """Return all retained samples, optionally repeated."""
        if duplicates < 1:
            raise ValueError("duplicates must be at least 1")

        if not self._batches:
            raise ValueError("cannot build dataset from empty replay buffer")

        total_data = {
            key: jnp.concatenate(
                [batch[key] for batch in self._batches],
                axis=0,
            )
            for key in self._keys
        }

        if duplicates == 1:
            return total_data

        return {
            key: jnp.tile(value, (duplicates,) + (1,) * (value.ndim - 1))
            for key, value in total_data.items()
        }

    def __len__(self):
        """Return the number of retained samples."""
        if not self._batches:
            return 0

        return sum(len(batch[self._keys[0]]) for batch in self._batches)

    def state_dict(self):
        """Return replay-buffer state."""
        return {
            "batches": self._batches,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state_dict):
        """Restore replay-buffer state."""
        self.capacity = state_dict["capacity"]
        self._batches = state_dict["batches"]

        if self._batches:
            self._keys = tuple(self._batches[0].keys())


class _FourierTimeEmbedding(nnx.Module):
    """Official repository-style trainable Fourier time embedding."""

    def __init__(
        self,
        channels: int,
        rngs: nnx.Rngs,
        phase_key: ArrayLike,
    ):
        super().__init__()

        self.channels = channels
        self.timestep_coeff = jnp.linspace(
            0.1,
            100.0,
            channels,
            dtype=jnp.float32,
        )[None, :]
        self.timestep_phase = nnx.Param(
            jax.random.normal(
                phase_key,
                (1, channels),
                dtype=jnp.float32,
            )
        )
        self.hidden_layer = nnx.Linear(
            2 * channels,
            channels,
            rngs=rngs,
        )
        self.out_layer = nnx.Linear(
            channels,
            channels,
            rngs=rngs,
        )

    def __call__(self, t: ArrayLike):
        """Embed scalar or batched normalized times."""
        t = jnp.asarray(t, dtype=jnp.float32)

        if t.ndim == 0:
            t = t[None, None]
        elif t.ndim == 1:
            t = t[:, None]

        phase = self.timestep_coeff * t + self.timestep_phase
        embedding = jnp.concatenate(
            [
                jnp.sin(phase),
                jnp.cos(phase),
            ],
            axis=-1,
        )
        embedding = nnx.gelu(self.hidden_layer(embedding))
        return self.out_layer(embedding)


class _FourierMLP(nnx.Module):
    """Official low-dimensional repository Fourier MLP."""

    def __init__(
        self,
        dim: int,
        channels: int = 64,
        num_layers: int = 4,
        seed: int = 0,
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be at least 1")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.dim = dim
        self.channels = channels
        self.num_layers = num_layers

        rngs = nnx.Rngs(seed)
        phase_key = jax.random.fold_in(
            jax.random.PRNGKey(seed),
            1,
        )

        self.input_embed = nnx.Linear(
            dim,
            channels,
            rngs=rngs,
        )
        self.time_embed = _FourierTimeEmbedding(
            channels=channels,
            rngs=rngs,
            phase_key=phase_key,
        )

        hidden_components = []
        for _ in range(num_layers - 2):
            hidden_components.extend(
                [
                    nnx.gelu,
                    nnx.Linear(
                        channels,
                        channels,
                        rngs=rngs,
                    ),
                ]
            )

        self.hidden_layers = nnx.Sequential(*hidden_components)
        self.out_layer = nnx.Linear(
            channels,
            dim,
            kernel_init=nnx.initializers.constant(0.0),
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(
        self,
        x: ArrayLike,
        t: ArrayLike,
    ):
        """Evaluate the time-dependent Fourier MLP."""
        x = jnp.asarray(x)
        t = jnp.asarray(t, dtype=jnp.float32)

        if t.ndim == 0:
            t = jnp.broadcast_to(
                t,
                (x.shape[0], 1),
            )
        elif t.ndim == 1:
            t = t[:, None]
        elif t.shape[0] == 1 and x.shape[0] != 1:
            t = jnp.broadcast_to(
                t,
                (x.shape[0], 1),
            )

        if t.shape != (x.shape[0], 1):
            raise ValueError("t must be scalar or have shape (batch_size, 1)")

        embedding = self.input_embed(x) + self.time_embed(t)

        if self.num_layers > 2:
            embedding = self.hidden_layers(embedding)

        return self.out_layer(nnx.gelu(embedding))


class _StaticCorrector(nnx.Module):
    """Static wrapper around the existing RMC MLP."""

    def __init__(self, config: NNConfigDict):
        super().__init__()

        from rmc.flax.blocks import MLP

        self.nn = MLP(
            ndim_in=config["dim"],
            ndim_out=config["dim"],
            layer_widths=config["layer_widths"],
            activation_func=config["activation_func"],
            zero_init_output=True,
            rngs=nnx.Rngs(config["seed"]),
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the terminal corrector."""
        return self.nn(x)

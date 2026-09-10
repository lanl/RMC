# -*- coding: utf-8 -*-
# Copyright (C) 2025-2026 by RMC Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the RMC package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Utilities for deploying an Adjoint Sampler."""

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
    train_step,
)
from rmc.utils.schedule_diffusion import (
    constant_diffusion_schedule,
    constant_integrated_variance,
    geometric_diffusion_schedule,
    geometric_integrated_variance,
)


class _ReplayBuffer:
    """Unbounded replay buffer for endpoint/terminal-adjoint pairs."""

    def __init__(
        self,
        dim: int,
        capacity: int | None = None,
    ):
        self.dim = dim
        self.capacity = capacity

        if capacity is not None and capacity < 1:
            raise ValueError("Replay-buffer capacity must be positive or None")

        self.endpoints = None
        self.terminal_gradients = None

    def __len__(self):
        if self.endpoints is None:
            return 0

        return self.endpoints.shape[0]

    def add(
        self,
        endpoints: ArrayLike,
        terminal_gradients: ArrayLike,
    ):
        """Append endpoint-gradient pairs to the buffer."""
        endpoints = jax.lax.stop_gradient(jnp.asarray(endpoints))
        terminal_gradients = jax.lax.stop_gradient(jnp.asarray(terminal_gradients))

        if endpoints.ndim != 2 or endpoints.shape[1] != self.dim:
            raise ValueError("endpoints must have shape (N, dim)")

        if terminal_gradients.shape != endpoints.shape:
            raise ValueError("terminal_gradients must have the same shape as endpoints")

        if self.endpoints is None:
            self.endpoints = endpoints
            self.terminal_gradients = terminal_gradients
        else:
            self.endpoints = jnp.concatenate(
                [self.endpoints, endpoints],
                axis=0,
            )
            self.terminal_gradients = jnp.concatenate(
                [self.terminal_gradients, terminal_gradients],
                axis=0,
            )

        if self.capacity is not None and len(self) > self.capacity:
            self.endpoints = self.endpoints[-self.capacity :]
            self.terminal_gradients = self.terminal_gradients[-self.capacity :]

    def sample(
        self,
        subkey: ArrayLike,
        batch_size: int,
    ):
        """Sample endpoint-gradient pairs uniformly with replacement."""
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")

        indices = jax.random.randint(
            subkey,
            (batch_size,),
            minval=0,
            maxval=len(self),
        )

        return (
            self.endpoints[indices],
            self.terminal_gradients[indices],
        )


class AdjointSampler(nnx.Module):
    """Adjoint Sampling with a scalar-diffusion Brownian reference."""

    def __init__(
        self,
        config: NNConfigDict,
        densitycl,
        h: float,
        T: int,
        sigma_schedule: Callable | None = None,
        integrated_variance: Callable | None = None,
        verbose: bool = False,
    ):
        """Initialize the Adjoint Sampler."""
        super().__init__()

        self.config = config
        self.Dcl = densitycl
        self.d = config["dim"]

        self.h = h
        self.T = T
        self.TT = h * T

        self.verbose = verbose

        (
            self.sigma_schedule,
            self.integrated_variance,
        ) = self._resolve_diffusion_schedule(
            sigma_schedule,
            integrated_variance,
        )

        self.base_terminal_variance = jnp.asarray(self.integrated_variance(self.TT))

        if float(self.base_terminal_variance) <= 0.0:
            raise ValueError("The terminal accumulated diffusion variance must be positive.")

        # Adjoint Sampling learns u_theta(x,t) directly.
        if config["nn_type"] == "time_embed":
            self.nnmodel = NN_with_time_embedding(self.config)
        elif config["nn_type"] == "score":
            raise ValueError("Adjoint Sampling does not use the score-informed RMC network")
        else:
            self.nnmodel = NN_with_time(self.config)

    def _repo_enabled(self):
        """Return whether repository-style practical features are enabled."""
        return bool(self.config.get("adjoint_repo_features", False))

    def _resolve_diffusion_schedule(
        self,
        sigma_schedule,
        integrated_variance,
    ):
        """Resolve explicit or config-defined scalar diffusion schedules."""
        if (sigma_schedule is None) != (integrated_variance is None):
            raise ValueError(
                "sigma_schedule and integrated_variance must either both "
                "be supplied or both be omitted"
            )

        if sigma_schedule is not None:
            return sigma_schedule, integrated_variance

        default_name = "geometric" if self._repo_enabled() else "constant"
        schedule_name = self.config.get(
            "adjoint_diffusion_schedule",
            default_name,
        )

        if schedule_name == "constant":
            sigma = float(
                self.config.get(
                    "adjoint_sigma",
                    1.0,
                )
            )

            if sigma <= 0.0:
                raise ValueError("adjoint_sigma must be positive")

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
            sigma_min = float(
                self.config.get(
                    "adjoint_sigma_min",
                    1.0e-3,
                )
            )
            sigma_max = float(
                self.config.get(
                    "adjoint_sigma_max",
                    1.0,
                )
            )

            if sigma_min <= 0.0 or sigma_max <= sigma_min:
                raise ValueError(
                    "Geometric diffusion requires " "0 < adjoint_sigma_min < adjoint_sigma_max"
                )

            return (
                partial(
                    geometric_diffusion_schedule,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    terminal_time=self.TT,
                ),
                partial(
                    geometric_integrated_variance,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    terminal_time=self.TT,
                ),
            )

        raise ValueError(f"Unsupported Adjoint diffusion schedule: {schedule_name}")

    def _resolve_training_options(self):
        """Resolve naive or repository-style training defaults once."""
        repo = self._repo_enabled()

        options = {
            "outer_iterations": int(
                self.config.get(
                    "adjoint_outer_iterations",
                    self.config.get("max_subiter", 1),
                )
            ),
            "outer_samples": int(
                self.config.get(
                    "adjoint_outer_samples",
                    self.config.get(
                        "max_samples",
                        self.config.get("nsamples", 1),
                    ),
                )
            ),
            "inner_steps": int(
                self.config.get(
                    "adjoint_inner_steps",
                    100 if repo else 1,
                )
            ),
            "batch_size": int(
                self.config.get(
                    "adjoint_batch_size",
                    self.config.get(
                        "batch_size",
                        self.config.get(
                            "adjoint_outer_samples",
                            self.config.get("nsamples", 1),
                        ),
                    ),
                )
            ),
            "replay_capacity": self.config.get(
                "adjoint_replay_capacity",
                1000 if repo else None,
            ),
            "init_base_samples": int(
                self.config.get(
                    "adjoint_init_base_samples",
                    1024 if repo else 0,
                )
            ),
            "target_clip": self.config.get(
                "adjoint_target_clip",
                150.0 if repo else None,
            ),
            "time_discretization": self.config.get(
                "adjoint_time_discretization",
                "ql" if repo else "uniform",
            ),
        }

        if options["replay_capacity"] is not None:
            options["replay_capacity"] = int(options["replay_capacity"])

        if options["target_clip"] is not None:
            options["target_clip"] = float(options["target_clip"])

        if options["outer_iterations"] < 1:
            raise ValueError("adjoint_outer_iterations must be at least 1")
        if options["outer_samples"] < 1:
            raise ValueError("adjoint_outer_samples must be at least 1")
        if options["inner_steps"] < 1:
            raise ValueError("adjoint_inner_steps must be at least 1")
        if options["batch_size"] < 1:
            raise ValueError("adjoint_batch_size must be at least 1")
        if options["init_base_samples"] < 0:
            raise ValueError("adjoint_init_base_samples must be nonnegative")
        if options["replay_capacity"] is not None and options["replay_capacity"] < 1:
            raise ValueError("adjoint_replay_capacity must be positive or None")
        if options["target_clip"] is not None and options["target_clip"] <= 0.0:
            raise ValueError("adjoint_target_clip must be positive or None")
        if options["time_discretization"] not in (
            "uniform",
            "ql",
        ):
            raise ValueError("adjoint_time_discretization must be 'uniform' or 'ql'")

        return options

    def _build_time_grid(self):
        """Build the Euler--Maruyama time grid."""
        scheme = self.config.get(
            "adjoint_time_discretization",
            "ql" if self._repo_enabled() else "uniform",
        )

        if scheme == "uniform":
            return jnp.linspace(
                0.0,
                self.TT,
                self.T + 1,
            )

        if scheme != "ql":
            raise ValueError(f"Unsupported Adjoint time discretization: {scheme}")

        if self.T < 3:
            raise ValueError("ql discretization requires at least 3 integration steps")

        # RMC implementation of the quadratic-linear grid used in the
        # released Cartesian Adjoint Sampling experiments.  It concentrates
        # integration points nonuniformly while preserving t=0 and t=TT.
        total = self.T + 2
        n_linear = int(0.5 * total)
        n_quadratic = self.T + 1 - n_linear

        reference = jnp.linspace(
            0.0,
            1.0,
            1000,
        )
        linear_part = reference[:n_linear]
        anchor = reference[n_linear]

        z = jnp.linspace(
            0.0,
            1.0,
            n_quadratic,
        )
        quadratic_part = anchor + (1.0 - anchor) * z**2

        reverse_parameter = jnp.concatenate(
            [
                linear_part,
                quadratic_part,
            ]
        )

        return self.TT * (1.0 - reverse_parameter)[::-1]

    def eval_sigma(self, t: ArrayLike):
        """Evaluate the scalar diffusion coefficient sigma(t)."""
        return self.sigma_schedule(t)

    def eval_integrated_variance(self, t: ArrayLike):
        """Evaluate the accumulated reference variance Q(t)."""
        return self.integrated_variance(t)

    def eval_terminal_gradient(self, x: ArrayLike):
        r"""Evaluate the terminal adjoint gradient.

        For

            g(x) = log p_base,T(x) - log target(x),

        with p_base,T = N(0, Q(T) I),

            grad g(x) = -x / Q(T) - grad log target(x).
        """
        x = jnp.asarray(x)

        target_score = self.Dcl.der_log_target_proposal(
            x,
            tempering=1.0,
        )

        return -x / self.base_terminal_variance - target_score

    def eval_bridge_moments(self, t: ArrayLike):
        r"""Evaluate the exact Brownian-reference bridge moments."""
        q_t = jnp.asarray(self.integrated_variance(t))

        alpha = q_t / self.base_terminal_variance
        variance = q_t * (1.0 - alpha)

        # Suppress tiny negative roundoff at the terminal endpoint.
        variance = jnp.maximum(variance, 0.0)

        return alpha, variance

    def sample_base_bridge(
        self,
        x_terminal: ArrayLike,
        t: ArrayLike,
        noise: ArrayLike,
    ):
        """Sample from the exact endpoint-conditioned base bridge."""
        x_terminal = jnp.asarray(x_terminal)
        noise = jnp.asarray(noise)

        alpha, variance = self.eval_bridge_moments(t)

        while alpha.ndim < x_terminal.ndim:
            alpha = alpha[..., None]
            variance = variance[..., None]

        return alpha * x_terminal + jnp.sqrt(variance) * noise

    def _format_network_time(
        self,
        t: ArrayLike,
        batch_size: int,
    ):
        """Format scalar or batched times for the RMC time networks."""
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
        """Evaluate the learned control u_theta(x,t)."""
        x = jnp.asarray(x)

        nn_time = self._format_network_time(
            t,
            x.shape[0],
        )

        return self.nnmodel(x, nn_time)

    def eval_drift(
        self,
        x: ArrayLike,
        t: ArrayLike,
    ):
        r"""Evaluate the controlled drift sigma(t) u_theta(x,t)."""
        control = self.eval_control(x, t)
        sigma = jnp.asarray(self.eval_sigma(t))

        while sigma.ndim < control.ndim:
            sigma = sigma[..., None]

        return sigma * control

    def generate_paths(
        self,
        nsamples: int,
        subkey: ArrayLike,
    ):
        r"""Generate controlled Euler--Maruyama sample paths.

        The controlled process is

            dX_t
                = sigma(t) u_theta(X_t,t) dt
                  + sigma(t) dW_t,

        with X_0 = 0.
        """
        key = subkey

        x = jnp.zeros(
            (nsamples, self.d),
            dtype=jnp.float32,
        )
        xpath = [x]

        self.nnmodel.eval()

        times = self._build_time_grid()

        for t, t_next in zip(
            times[:-1],
            times[1:],
        ):
            dt = t_next - t

            drift = self.eval_drift(
                x,
                t,
            )
            sigma = self.eval_sigma(t)

            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(
                noise_key,
                (nsamples, self.d),
                dtype=x.dtype,
            )

            x = x + dt * drift + sigma * jnp.sqrt(dt) * noise

            xpath.append(x)

        return xpath

    def generate_endpoints(
        self,
        nsamples: int,
        subkey: ArrayLike,
    ):
        """Generate terminal states from the current controlled process."""
        return self.generate_paths(nsamples, subkey)[-1]

    def sample_base_endpoints(
        self,
        nsamples: int,
        subkey: ArrayLike,
    ):
        """Sample terminal endpoints exactly from the uncontrolled base law."""
        noise = jax.random.normal(
            subkey,
            (nsamples, self.d),
            dtype=jnp.float32,
        )

        return jnp.sqrt(self.base_terminal_variance) * noise

    def _clip_terminal_gradients(
        self,
        terminal_gradients: ArrayLike,
        max_norm: float | None,
    ):
        """Clip cached terminal-adjoint vectors by Euclidean norm."""
        terminal_gradients = jnp.asarray(terminal_gradients)

        if max_norm is None:
            return terminal_gradients, 0.0

        norms = jnp.linalg.norm(
            terminal_gradients,
            axis=-1,
        )

        coefficients = jnp.minimum(
            1.0,
            max_norm / (norms + 1.0e-6),
        )

        clipped = terminal_gradients * coefficients[:, None]

        fraction_clipped = jnp.mean(norms > max_norm)

        return clipped, fraction_clipped

    def build_ram_batch(
        self,
        subkey: ArrayLike,
        endpoints: ArrayLike,
        terminal_gradients: ArrayLike,
        batch_size: int | None = None,
    ):
        r"""Build a reciprocal-adjoint-matching regression batch.

        For a stored endpoint/terminal-gradient pair (Y,G), sample

            t ~ Uniform(0,T),

            X_t ~ p_base(X_t | X_T=Y),

        and use the regression target

            u_target = -sigma(t) G.

        If ``batch_size`` is supplied, endpoint-gradient pairs are sampled
        uniformly with replacement from the supplied arrays.
        """
        endpoints = jnp.asarray(endpoints)
        terminal_gradients = jnp.asarray(terminal_gradients)

        if endpoints.ndim != 2:
            raise ValueError("endpoints must have shape (N, dim)")

        if terminal_gradients.shape != endpoints.shape:
            raise ValueError("terminal_gradients must have the same shape as endpoints")

        npoints = endpoints.shape[0]

        if npoints == 0:
            raise ValueError("Cannot build a RAM batch from an empty array")

        index_key, time_key, noise_key = jax.random.split(
            subkey,
            3,
        )

        if batch_size is None:
            batch_size = npoints
            selected_endpoints = endpoints
            selected_gradients = terminal_gradients
        else:
            indices = jax.random.randint(
                index_key,
                (batch_size,),
                minval=0,
                maxval=npoints,
            )
            selected_endpoints = endpoints[indices]
            selected_gradients = terminal_gradients[indices]

        times = jax.random.uniform(
            time_key,
            (batch_size,),
            minval=0.0,
            maxval=self.TT,
            dtype=endpoints.dtype,
        )

        noise = jax.random.normal(
            noise_key,
            selected_endpoints.shape,
            dtype=endpoints.dtype,
        )

        bridge_samples = self.sample_base_bridge(
            selected_endpoints,
            times,
            noise,
        )

        sigma = jnp.asarray(self.eval_sigma(times))
        while sigma.ndim < selected_gradients.ndim:
            sigma = sigma[..., None]

        targets = -sigma * selected_gradients

        train_input = jnp.concatenate(
            [
                bridge_samples,
                times[:, None],
            ],
            axis=-1,
        )

        return {
            "input": train_input,
            "label": targets,
        }

    def compute_ram_loss(
        self,
        model,
        input: ArrayLike,
        labels: ArrayLike,
    ):
        r"""Evaluate the reciprocal adjoint matching regression loss.

        With

            u_target = -sigma(t) grad g(X_T),

        the reciprocal-adjoint-matching objective is

            1/2 E[
                ||u_theta(X_t,t) - u_target||^2 / sigma(t)^2
            ].
        """
        input = jnp.asarray(input)
        labels = jnp.asarray(labels)

        x = input[:, : self.d]
        t = input[:, self.d :]

        output = model(x, t)

        sigma = jnp.asarray(self.eval_sigma(t.squeeze(-1)))
        sigma = sigma[:, None]

        residual = (output - labels) / sigma

        return 0.5 * jnp.mean(
            jnp.sum(
                residual**2,
                axis=-1,
            )
        )

    def sample(
        self,
        nsamples: int,
        subkey: ArrayLike,
    ):
        """Generate complete sample paths from the learned controlled SDE."""
        return self.generate_paths(
            nsamples,
            subkey,
        )

    def _build_optimizer(self):
        """Build the persistent optimizer used by Adjoint Sampling."""
        if "lr_schedule" in self.config:
            learning_rate_fn = self.config["lr_schedule"]
        else:
            learning_rate_fn = optax.constant_schedule(self.config["base_lr"])

        tx = build_optax_optimizer(
            self.config,
            learning_rate_fn,
        )

        return nnx.Optimizer(
            self.nnmodel,
            tx,
            wrt=nnx.Param,
        )

    def train(self):
        r"""Train the control using reciprocal adjoint matching.

        The common algorithm is

            rollout
                -> terminal adjoint evaluation
                -> replay insertion
                -> repeated exact-base-bridge RAM updates.

        ``adjoint_repo_features=True`` changes practical defaults while
        preserving this same core training path.
        """
        options = self._resolve_training_options()

        nouter = options["outer_iterations"]
        nnew = options["outer_samples"]
        ninner = options["inner_steps"]
        batch_size = options["batch_size"]
        target_clip = options["target_clip"]

        optimizer = self._build_optimizer()

        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

        replay = _ReplayBuffer(
            self.d,
            capacity=options["replay_capacity"],
        )

        key = jax.random.PRNGKey(self.config["seed"])

        history = []

        # Repository-style training starts from exact uncontrolled-base
        # endpoints.  Their target gradients are cached exactly once.
        if options["init_base_samples"] > 0:
            key, base_key = jax.random.split(key)

            base_endpoints = self.sample_base_endpoints(
                options["init_base_samples"],
                base_key,
            )

            base_gradients = self.eval_terminal_gradient(base_endpoints)

            base_gradients, _ = self._clip_terminal_gradients(
                base_gradients,
                target_clip,
            )

            replay.add(
                base_endpoints,
                base_gradients,
            )

        log_every = int(
            self.config.get(
                "adjoint_log_every",
                self.config.get("eval_every", 1),
            )
        )
        log_every = max(log_every, 1)

        for outer in range(nouter):
            (
                key,
                path_key,
                ram_key,
            ) = jax.random.split(
                key,
                3,
            )

            endpoints = self.generate_endpoints(
                nnew,
                path_key,
            )

            terminal_gradients = self.eval_terminal_gradient(endpoints)

            (
                terminal_gradients,
                fraction_clipped,
            ) = self._clip_terminal_gradients(
                terminal_gradients,
                target_clip,
            )

            replay.add(
                endpoints,
                terminal_gradients,
            )

            inner_losses = []

            for _ in range(ninner):
                (
                    ram_key,
                    replay_key,
                    bridge_key,
                ) = jax.random.split(
                    ram_key,
                    3,
                )

                (
                    replay_endpoints,
                    replay_gradients,
                ) = replay.sample(
                    replay_key,
                    batch_size,
                )

                train_ds = self.build_ram_batch(
                    bridge_key,
                    replay_endpoints,
                    replay_gradients,
                    batch_size=None,
                )

                self.nnmodel.train()

                loss = train_step(
                    self.nnmodel,
                    self.compute_ram_loss,
                    optimizer,
                    metrics,
                    train_ds["input"],
                    train_ds["label"],
                    False,
                )

                metrics.reset()
                inner_losses.append(loss)

            inner_losses = jnp.asarray(inner_losses)

            gradient_norms = jnp.linalg.norm(
                terminal_gradients,
                axis=-1,
            )

            entry = {
                "loss": float(jnp.mean(inner_losses)),
                "buffer_size": len(replay),
                "terminal_gradient_norm_mean": float(jnp.mean(gradient_norms)),
                "terminal_gradient_norm_max": float(jnp.max(gradient_norms)),
                "terminal_gradient_clip_fraction": float(fraction_clipped),
            }
            history.append(entry)

            if outer == 0 or (outer + 1) % log_every == 0 or outer + 1 == nouter:
                print(
                    f"Adjoint outer {outer + 1}/{nouter} --> "
                    f"loss: {entry['loss']:.6e}, "
                    f"buffer: {entry['buffer_size']}, "
                    f"mean |grad g|: "
                    f"{entry['terminal_gradient_norm_mean']:.6e}, "
                    f"clipped: "
                    f"{entry['terminal_gradient_clip_fraction']:.3f}"
                )

        if "root_path" in self.config:
            save_model(
                self.nnmodel,
                self.config["root_path"],
                "nnx-state-adjoint",
            )

        return history

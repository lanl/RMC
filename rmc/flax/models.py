# -*- coding: utf-8 -*-
# Copyright (C) 2025 by RMC Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the RMC package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construction of Flax Neural Networks."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from .blocks import MLP, SinusoidalPositionEmbeddings
from .nn_config_dict import NNConfigDict


class NN_with_time_embedding(nnx.Module):
    """Definition of neural network model with time dependence via
    sinusoidal embedding."""

    def __init__(
        self,
        config: NNConfigDict,
        dim_sine_embedding=128,
        zero_init_output: bool = False,
    ):
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
            zero_init_output=zero_init_output,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        """Evaluate control policy.

        Args:
            x: The position array to be evaluated.
            t: The times to be evaluated.

        Returns:
            Control policy at current samples.
        """
        if isinstance(t, float) or isinstance(t, int):
            t = jnp.tile(self.time_mlp(t), (x.shape[0], 1))
        else:
            t = self.time_mlp(t)
        x_t = jnp.concatenate([x, t], axis=-1)

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

    def __call__(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        """Evaluate control policy.

        Args:
            x: The position array to be evaluated.
            t: The times to be evaluated.

        Returns:
            Control policy at current samples.
        """
        if isinstance(t, float) or isinstance(t, int):
            t = jnp.tile(jnp.asarray(t, dtype=jnp.float32), (x.shape[0], 1))
        x_t = jnp.concatenate([x, t], axis=-1)

        return self.nn(x_t)


class NN_gradient_informed(nnx.Module):
    """Definition of policy model with two neural networks.

    The model includes one NN with time and spatial dependence
    and another NN with time dependence multiplying the score function."""

    def __init__(self, config: NNConfigDict, score_fn: Callable):
        super().__init__()
        rngs = nnx.Rngs(config["seed"])
        dim = config["dim"]

        self.score_weight_mode = config.get("score_weight_mode", "scalar")
        self.stop_score_gradient = config.get("stop_score_gradient", False)
        self.score_clip = config.get("score_clip", None)
        self.state_output_clip = config.get("state_output_clip", None)

        if self.score_clip is not None and self.score_clip <= 0:
            raise ValueError("score_clip must be positive or None.")
        if self.state_output_clip is not None and self.state_output_clip <= 0:
            raise ValueError("state_output_clip must be positive or None.")

        if self.score_weight_mode == "scalar":
            score_weight_dim = 1
        elif self.score_weight_mode == "vector":
            score_weight_dim = dim
        else:
            raise ValueError(
                "Unsupported score_weight_mode "
                f"{self.score_weight_mode!r}. Expected 'scalar' or 'vector'."
            )

        # NN with time and spatial dependence
        self.nn1 = NN_with_time_embedding(
            config,
            zero_init_output=config.get("zero_init_output", False),
        )

        # NN with time dependence multiplying the target score
        self.nn2 = MLP(
            ndim_in=1,
            ndim_out=score_weight_dim,
            layer_widths=config["layer_widths_t"],
            activation_func=config["activation_func"],
            zero_init_output=config.get("zero_init_score_weight", False),
            rngs=rngs,
        )

        self.score_fn = score_fn

    def __call__(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        """Evaluate control policy.

        Args:
            x: The position array to be evaluated.
            t: The times to be evaluated.

        Returns:
            Control policy at current samples.
        """
        if isinstance(t, float) or isinstance(t, int):
            t = jnp.tile(jnp.asarray(t, dtype=jnp.float32), (x.shape[0], 1))

        state = self.nn1(x, t)
        if self.state_output_clip is not None:
            state = jnp.clip(
                state,
                -self.state_output_clip,
                self.state_output_clip,
            )

        score = self.score_fn(x)
        if self.stop_score_gradient:
            score = jax.lax.stop_gradient(score)
        if self.score_clip is not None:
            score = jnp.clip(
                score,
                -self.score_clip,
                self.score_clip,
            )

        return state + self.nn2(t) * score

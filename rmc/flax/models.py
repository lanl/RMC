# -*- coding: utf-8 -*-
# Copyright (C) 2025 by RMC Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the RMC package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construction of Flax Neural Networks."""


import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx

from .blocks import MLP, SinusoidalPositionEmbeddings
from .nn_config_dict import NNConfigDict


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
        """Evaluate control policy.

        Args:
            x: The position array to be evaluated.
            t: The time to be evaluated.

        Returns:
            Control policy at current samples.
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
        """Evaluate control policy.

        Args:
            x: The position array to be evaluated.
            t: The time to be evaluated.

        Returns:
            Control policy at current samples.
        """
        t_ = jnp.tile(jnp.asarray(t, dtype=jnp.float32), (x.shape[0], 1))
        x_t = jnp.concatenate([x, t_], axis=-1)

        return self.nn(x_t)

# -*- coding: utf-8 -*-
# Copyright (C) 2025 by RMC Developers
# All rights reserved. BSD 3-clause License.
# This file is part of the RMC package. Details of the copyright and
# user license can be found in the 'LICENSE' file distributed with the
# package.

"""Construction of Flax Neural Networks."""

from typing import Callable, Sequence

import jax.numpy as jnp
from jax.typing import ArrayLike

from flax import nnx


class MLP(nnx.Module):
    """Multi-layer perceptron (MLP) model."""

    def __init__(
        self,
        ndim_in: int,
        ndim_out: int,
        layer_widths: Sequence[int],
        activation_func: Callable = nnx.relu,
        activate_final: bool = False,
        batch_norm: bool = False,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """Initialization of MLP model.

        Args:
            ndim_in: Dimension of input.
            ndim_out: Dimension of output.
            layer_widths: Sequence of neurons per layer in MLP.
            activation_func: Activation function.
            activate_final: Flag to indicate if the activation function is
                to be applied after the final layer or not.
            batch_norm: Flag to indicate if batch norm is to be applied or not.
            rngs: Random generation key.
        """
        super().__init__()
        # Store model parameters
        self.ndim_in = ndim_in
        self.ndim_out = ndim_out
        self.activate_final = activate_final
        self.activation_func = activation_func

        # Declare layers
        if batch_norm:
            self.layers = nnx.Sequential(
                nnx.Linear(in_features=ndim_in, out_features=layer_widths[0], rngs=rngs),
                *[
                    nnx.Sequential(
                        nnx.BatchNorm(layer_widths[i], rngs=rngs),
                        activation_func,
                        nnx.Linear(in_features=layer_widths[i], out_features=lyw, rngs=rngs),
                    )
                    for i, lyw in enumerate(layer_widths[1:])
                ],
                nnx.BatchNorm(layer_widths[-1], rngs=rngs),
                activation_func,
                nnx.Linear(
                    in_features=layer_widths[-1],
                    out_features=ndim_out,
                    kernel_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                ),
            )
        else:
            self.layers = nnx.Sequential(
                nnx.Linear(in_features=ndim_in, out_features=layer_widths[0], rngs=rngs),
                *[
                    nnx.Sequential(
                        activation_func,
                        nnx.Linear(in_features=layer_widths[i], out_features=lyw, rngs=rngs),
                    )
                    for i, lyw in enumerate(layer_widths[1:])
                ],
                activation_func,
                # nnx.Linear(in_features=layer_widths[-1], out_features=ndim_out, kernel_init=nnx.initializers.constant(0.), rngs=rngs)
                nnx.Linear(in_features=layer_widths[-1], out_features=ndim_out, rngs=rngs),
            )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Apply fully connected (i.e. dense) layer(s), batch norm (optional), dropout (optional) and activation(s).

        Args:
            x: The array to be transformed.

        Returns:
            The input after being transformed by the multiple layers
            of the MLP.
        """
        x = self.layers(x)
        if self.activate_final:
            x = self.activation_func(x)
        return x


def get_timestep_embedding(timesteps: ArrayLike, embedding_dim: int = 128):
    """Construct an embedding for a sequence of time steps.

    Args:
        timesteps: Sequence of time steps to embed.
        embedding_dim: Embedding dimension.

    Returns:
        Time steps as an embedded sequence with specified dimension.
    """
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)

    emb = jnp.asarray(timesteps, dtype=jnp.float32) * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = jnp.pad(emb, [0, 1], mode="constant")

    return emb


class SinusoidalPositionEmbeddings(nnx.Module):
    """Define sinusoidal positional embeddings class."""

    def __init__(self, dim: int):
        """Initialize sinusoidal position embeddings class.

        Args:
            dim: Embedding dimension.
        """
        super().__init__()
        self.dim = dim

    def __call__(self, time):
        """Compute embeddings."""
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -embeddings)
        embeddings = jnp.asarray(time, dtype=jnp.float32) * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings

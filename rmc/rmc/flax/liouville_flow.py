# -*- coding: utf-8 -*-

"""Utilities for deploying a Liouville flow-based sampler."""

from typing import Callable, Sequence

from jax.typing import ArrayLike

from flax import nnx

from .nn_config_dict import NNConfigDict

class LiouvilleFlow(nnx.Module):
    """Definition of Liouville Flow class."""
    def __init__(self, config: NNConfigDict):
        """Initialization of Liouville Flow class.
        
        Args:
            config: Dictionary with configuration parameters.
        """
        super().__init__()
        # Store model parameters

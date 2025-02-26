# -*- coding: utf-8 -*-

"""Definition of typed dictionaries for objects in sampling functionality."""

import sys
from typing import Callable, Tuple

from jax.typing import ArrayLike

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


RealArray = ArrayLike

class ConfigDict(TypedDict):
    """Dictionary structure for sampler parameters.

    Definition of the dictionary structure expected for specifying
    sampler parameters."""

    seed: float
    batch_size: int
    sample_shape: Tuple[int]
    initial_sampler_fn: Callable
    initial_sampler_mean: RealArray
    initial_sampler_covariance: RealArray
    maxiter: int
    numsteps: int
    log_freq: int
    energy_cl: Callable
    ESS_thres: float

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
    numleapfrog: int
    log_freq: int
    energy_cl: Callable
    step_size: float
    store_path: bool
    ESS_thres: float
    tempering_fn: Callable

# -*- coding: utf-8 -*-

"""Definition of typed dictionaries for objects in sampling functionality."""

import sys
from typing import Callable, Tuple

from jax.typing import ArrayLike

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict

from rmc.utils.packed_distributions import BasePackedDistribution

RealArray = ArrayLike


class ConfigDict(TypedDict):
    """Dictionary structure for sampler parameters.

    Definition of the dictionary structure expected for specifying
    sampler parameters."""

    #: Value to initialize seed for random generation.
    seed: float
    #: Size of batch for generation.
    batch_size: int
    #: Shape (i.e. dimension) of samples.
    sample_shape: Tuple[int]
    #: Object to use for generating initial samples
    initial_sampler_cl: BasePackedDistribution
    #: Number of iterations in current run for sampler
    maxiter: int
    #: Number of steps per iteration of sampler method
    numsteps: int
    #: Number of iterations of leapfrog method per step in HMC
    numleapfrog: int
    #: Frequency of logging monitored metrics
    log_freq: int
    #: Class representing target density function
    density_cl: Callable
    #: Size of step to advance in HMC
    step_size: float
    #: Flag to indicate if intermediate sample trajectories are to be stored
    store_path: bool
    #: Effective sampling size (ESS) metric
    ESS_thres: float
    #: Function for density tempering
    tempering_fn: Callable

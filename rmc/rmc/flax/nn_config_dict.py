# -*- coding: utf-8 -*-

"""Definition of typed dictionaries for objects in neural network functionality."""

import sys
from typing import Callable, Sequence

from jax.typing import ArrayLike

if sys.version_info >= (3, 8):
    from typing import TypedDict  # pylint: disable=no-name-in-module
else:
    from typing_extensions import TypedDict


class NNConfigDict(TypedDict):
    """Dictionary structure for neural network parameters.

    Definition of the dictionary structure expected for specifying
    neural network architecture and training parameters."""

    seed: float
    task: str
    batch_size: int
    method: Callable
    layer_widths: Sequence
    opt_type: str
    base_lr: float
    lr_schedule: Callable
    patience: int
    criterion: Callable
    max_epochs: int



class DataSetDict(TypedDict):
    """Dictionary structure for training data sets.

    Definition of the dictionary structure
    expected for the training data sets."""

    input: Array  # input
    label: Array  # output

# -*- coding: utf-8 -*-
"""Renovating Monte Carlo (RMC) is a Python package for sampling from
high-dimensional distributions via Monte Carlo-like algorithms as well
as advanced AI/ML-based generative models.
"""

__version__ = "0.0.1"

import sys

import jax

from .modules.lfis import LiouvilleFlow
from .modules.pis import PathIntegralSampler
from .modules.sampler import HMC, SMC
from .modules.svgd import SVGD
from .utils.config_dict import ConfigDict
from .utils.density import (
    BaseLogDensity,
    LinearRegressionDensity,
    LogDensityPath,
    LogDensityPosterior,
)
from .utils.packed_distributions import PackedMultivariateNormal, PackedNormal
from .utils.plotting import (
    plot_func_contours,
    plot_func_xDim_contours,
    plot_quiver,
    plot_samples,
    plot_trajectories,
    save_plot,
)
from .utils.schedule import CosineSchedule, LinearSchedule, QuadraticSchedule

# See https://github.com/google/jax/issues/19444
jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "ConfigDict",
    "BaseLogDensity",
    "LogDensityPath",
    "LogDensityPosterior",
    "HMC",
    "LinearRegressionDensity",
    "SMC",
    "SVGD",
    "LiouvilleFlow",
    "CosineSchedule",
    "LinearSchedule",
    "QuadraticSchedule",
    "PackedNormal",
    "PackedMultivariateNormal",
]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__

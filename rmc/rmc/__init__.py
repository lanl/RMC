# -*- coding: utf-8 -*-
name = "rmc"
__version__ = "0.0.1"

from .modules.sampler import HMC, SMC
from .modules.svgd import SVGD
from .modules.energy import LogDensity, LogDensityPath, LogPosterior, LinearRegressionE
from .utils.config_dict import ConfigDict

__all__ = [
    "ConfigDict",
    "LogDensity",
    "LogDensityPath",
    "LogPosterior",
    "HMC",
    "LinearRegressionE",
    "SMC",
    "SVGD",
]

# -*- coding: utf-8 -*-
__version__ = "0.0.1"
import sys

import jax

from .modules.sampler import HMC, SMC
from .modules.svgd import SVGD
from .modules.lfis import LiouvilleFlow
from .utils.energy import LogDensity, LogDensityPath, LogPosterior, LinearRegressionE
from .utils.config_dict import ConfigDict
from .utils.schedule import CosineSchedule, LinearSchedule

# See https://github.com/google/jax/issues/19444
jax.config.update("jax_default_matmul_precision", "highest")

__all__ = [
    "ConfigDict",
    "LogDensity",
    "LogDensityPath",
    "LogPosterior",
    "HMC",
    "LinearRegressionE",
    "SMC",
    "SVGD",
    "LiouvilleFlow",
    "CosineSchedule",
    "LinearSchedule",
]

# Imported items in __all__ appear to originate in top-level functional module
for name in __all__:
    getattr(sys.modules[__name__], name).__module__ = __name__

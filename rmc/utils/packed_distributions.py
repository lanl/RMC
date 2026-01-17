# -*- coding: utf-8 -*-

"""Utilities for sampling and getting statistics from common distributions"""

from abc import ABC, abstractmethod
from typing import Tuple

from jax.typing import ArrayLike


class BasePackedDistribution(ABC):
    """Base distribution class for sampling and getting statistics.

    A :class:`BasePackedDistribution` is the base class for packing
    existing distributions and enable a dual role of sampling and
    computing statistics (e.g. log probability density function).
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize variables and supplementary functions for
        distribution sampling and statistics evaluation (e.g.
        logpdf).

        Args:
            kwargs: Additional arguments that may be used by derivated
                classes.
        """
        raise NotImplementedError

    @abstractmethod
    def log_pdf(self, x: ArrayLike) -> ArrayLike:
        """Computation of the log probability density function (pdf).

        Args:
            x: Array of samples to evaluate log-pdf.

        Returns:
            Array of evaluated log-pdf.
        """
        raise NotImplementedError

    @abstractmethod
    def rvs(self, shape: Tuple[int], rngk: ArrayLike) -> ArrayLike:
        """Random variable sampling (rvs).

        Args:
            shape: Tuple describing the shape of samples to generate.
            rngk: Key for random number generation.

        Returns:
            Generated samples.
        """
        raise NotImplementedError

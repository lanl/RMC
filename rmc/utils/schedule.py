# -*- coding: utf-8 -*-

"""Utilities for defining time schedules for flows."""

import jax
import jax.numpy as jnp


class BaseSchedule:
    r"""Base class for defining time schedules.

    A schedule is a monotonic function :math:`\tau(t)`, transforming time :math:`t`, and
    satisfying :math:`\tau(0) = 0` and :math:`\tau(1) = 1`.
    """

    def __call__(self, t):
        """Evaluate schedule function.

        Args:
            t: Time to evaluate schedule function.

        Returns: Schedule function and derivative of schedule function evaluated at t.
        """
        outtau = self.tau(t)
        douttau = self.dtau(t)

        return outtau, douttau

    def tau(self, t):
        """Definition of schedule function.

        Args:
            t: Time to evaluate schedule function.

        Returns:
            Schedule function evaluated at t.
        """
        raise NotImplementedError

    def dtau(self, t):
        """Definition of derivative of schedule function.

        Args:
            t: Time to evaluate schedule function.

        Returns:
            Derivative of schedule function evaluated at t.
        """
        return jax.grad(self.tau)(t)


class CosineSchedule(BaseSchedule):
    """Class for defining a cosine schedule."""

    def tau(self, t):
        """Definition of cosine schedule function.

        Args:
            t: Time to evaluate cosine schedule function.

        Returns:
            Cosine schedule function evaluated at t.
        """
        return 0.5 * (1.0 - jnp.cos(jnp.pi * t))


class LinearSchedule(BaseSchedule):
    """Class for defining a linear schedule."""

    def tau(self, t):
        """Definition of linear schedule function.

        Args:
            t: Time to evaluate linear schedule function.

        Returns:
            Linear schedule function evaluated at t.
        """
        return t

# -*- coding: utf-8 -*-

"""Miscellaneous helper functions."""

from typing import Any


def exists(x: Any):
    """Determine if x is not none."""
    return x is not None


def default(val: Any, d: Any):
    """Return default value if given. Otherwise return object d.
    Args:
        val: Default value.
        d: Function or variable to return if no default value provided.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d

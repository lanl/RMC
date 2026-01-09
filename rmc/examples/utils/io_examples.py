#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define functionality for reading/preprocessing real data.
"""
import pickle

import numpy as np


def pad_with_const(x):
    """Pad array with row of ones.

    Args:
        x: Array to process.
    Returns:
        Processed array.
    """
    extra = np.ones((x.shape[0], 1))
    return np.hstack([extra, x])


def standardize_and_pad(x):
    """Standardize data and pad with constant value.

    Args:
        x: Array to process.
    Returns:
        Processed array.
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    return pad_with_const(x)


def load_data(name):
    """Load data from pickle file and preprocess.

    Args:
        name: Name of file to read.
    Returns:
        Arrays of (x, y) data read.
    """
    with open(name, mode="rb") as f:
        x, y = pickle.load(f)
    y = (y + 1) // 2
    x = standardize_and_pad(x)
    return x, y

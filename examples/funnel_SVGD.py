#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of SVGD for Funnel Density Function
===========================================

This script demonstrates the usage of a Stein variational gradient descent (SVGD)
sampler for sampling from the funnel density function.
"""


import os
import sys

import jax.numpy as jnp
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import FunnelDensity

from rmc import SVGD, ConfigDict, plot_func_xDim_contours, plot_samples

RealArray = ArrayLike

"""
Set distribution: 10D funnel distribution.
"""
d = 10  # Dimension of funnel distribution
x0_stddev = 3.0  # Standard deviation for component 0

# Define initial distribution
mean0 = 0.0  # initial mean
std0 = 1.0  # initial standard deviation
# Initial mean in array form
mean0_arr = mean0 * jnp.ones((1, d))
# Initial covariance matrix
cov0_arr = jnp.diagflat(std0**2 * jnp.ones((d,))).reshape((1, d, d))

Dcl = FunnelDensity(d, x0_stddev, mean0_arr, cov0_arr)


"""
Configure sampling run.
"""
# define sampling configuration
N = 2000  # 4000 #2000    # Number of particles

smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "maxiter": 1500,
    "log_freq": 100,
    "density_cl": Dcl,
    "step_size": 0.02,
    "kernel_parameter": 0.19,  # -1, # -1: Compute median
    "update_weight": 0.8,  # Alpha in code
}
print(f"Sampling configured --> parameters: {smp_conf}")

"""
Construct sampling object.
"""
svgd_obj = SVGD(N, smp_conf)
print("SVGD object constructed")

"""
Run sampler.
"""
samples = svgd_obj.sample()
print("Collected SVGD samples: ", samples.shape)

"""
Plot samples projected into (x0, x1) plane.
"""
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.size": 16})

# Plot funnel density contours
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
min, max = -10, 10
n = 120
keepscale = False  # Exponentiate log-target
cmap = "viridis"
cbar = True
ax = plot_func_xDim_contours(
    Dcl.log_target,
    d,
    ax,
    min,
    max,
    min,
    max,
    n,
    n,
    keepscale=keepscale,
    cbar=cbar,
    cmap=cmap,
    norm=LogNorm(),
    extend="both",
)
# Overlay SVGD results
ax = plot_samples(
    samples,
    ax,
    label="SVGD Samples",
    size=5,
    alpha=0.2,
    zorder=1,
    color="red",
)
# Display plots
plt.show()

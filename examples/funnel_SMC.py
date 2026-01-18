#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of SMC for Funnel Density Function
==========================================

This script demonstrates the usage of a sequential Monte Carlo (SMC)
sampler for sampling from the funnel density function.
"""


import os
import sys

import jax.numpy as jnp
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import FunnelDensity

from rmc import SMC, ConfigDict, plot_func_xDim_contours, plot_samples

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
# Define sampling configuration
N = 2000  # Number of particles
T = 256  # Number of tempering scales

# Define tempering
sched = jnp.linspace(0, 1, T + 1)
tempering_fn = lambda tstep: sched[tstep]

# Define configuration dictionary
prior_mean = 0.0  # prior mean
prior_std = 1.0  # prior standard deviation

smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "maxiter": T,
    "numsteps": 10,
    "numleapfrog": 20,
    "log_freq": 2,
    "density_cl": Dcl,
    "ESS_thres": 0.98,
    "step_size": 0.02,
    "tempering_fn": tempering_fn,
}
print(f"Sampling configured --> parameters: {smp_conf}")

"""
Construct sampling object.
"""
smc_obj = SMC(N, T, smp_conf)
print("SMC object constructed")

"""
Run sampler.
"""
smc_obj.sample()

samples = jnp.array(smc_obj.hmc_.qall)
print("Collected SMC samples: ", samples.shape)

"""
Plot samples for last T step projected into (x0, x1) plane.
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
# Overlay SMC results
ax = plot_samples(
    samples[-1],
    ax,
    label="SMC Samples",
    size=5,
    alpha=0.2,
    zorder=1,
    color="red",
)
# Display plots
plt.show()

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
from jax.random import multivariate_normal
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import FunnelDensity

from rmc import SMC, ConfigDict

RealArray = ArrayLike

"""
Set distribution: 10D funnel distribution.
"""
d = 10  # Dimension of funnel distribution
x0_stddev = 3.0  # Standard deviation for component 0
xg0_mean = 0.0  # Mean for components > 0
xg0_stddev = 1.0  # Standard deviation for components > 0
xg0_mean_vec = xg0_mean * jnp.ones((1, d))
xg0_stddev_vec = xg0_stddev * jnp.ones((1, d))

Dcl = FunnelDensity(d, x0_stddev, xg0_mean_vec, xg0_stddev_vec)


"""
Configure sampling run.
"""
# define sampling configuration
N = 2000  # Number of particles
T = 256  # Number of tempering scales

# define tempering
sched = jnp.linspace(0, 1, T + 1)
tempering_fn = lambda tstep: sched[tstep]
# cosine
# sched = (1.0 - jnp.cos(jnp.pi * jnp.linspace(0, 1, T + 1))) / 2.
# tempering_fn = lambda tstep : sched[tstep]

# define configuration dictionary
prior_mean = 0.0  # prior mean
prior_std = 1.0  # prior standard deviation

smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,))) ** 2).reshape(
        (1, d, d)
    ),
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

plt.rcParams.update({"font.size": 16})
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.scatter(samples[-1, :, 0], samples[-1, :, 1], s=5, marker="o", label="SMC samples", zorder=0)
ax.axis("equal")
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.legend(loc=2, frameon=False)
plt.show()

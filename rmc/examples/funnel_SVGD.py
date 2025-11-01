#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of SVGD for Funnel Distribution
=======================================

This script demonstrates the usage of a Stein variational gradient descent (SVGD)
sampler for sampling from the funnel distribution.
"""

from typing import Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jax.random import multivariate_normal

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rmc import ConfigDict, SVGD

from distributions_examples import FunnelE

RealArray = ArrayLike

"""
Set distribution: 10D funnel distribution.
"""
d = 10                  # Dimension of funnel distribution
x0_stddev = 3.          # Standard deviation for component 0
xg0_mean = 0.0          # Mean for components > 0
xg0_stddev = 1.0        # Standard deviation for components > 0
xg0_mean_vec = xg0_mean * jnp.ones((1, d))
xg0_stddev_vec = xg0_stddev * jnp.ones((1, d))

Ecl = FunnelE(d, x0_stddev, xg0_mean_vec, xg0_stddev_vec)


"""
Configure sampling run.
"""
# define sampling configuration
N = 2000 # 4000 #2000    # Number of particles

# define configuration dictionary
prior_mean = 0.0   # prior mean
prior_std = 1.0    # prior standard deviation

smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": 1500,
    "log_freq": 100,
    "energy_cl": Ecl,
    "step_size": 0.02,
    "kernel_parameter": 0.19, #-1, # -1: Compute median
    "update_weight": 0.8, # Alpha in code
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
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
fig,ax = plt.subplots(1, 1, figsize=(9,5))
ax.scatter(samples[:, 0], samples[:, 1], s=5, marker = 'o', label = 'SVGD samples', zorder=0)
ax.axis('equal')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.legend(loc=2,frameon=False)
plt.show()

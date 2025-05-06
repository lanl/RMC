#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of SMC for Funnel Distribution
======================================

This script demonstrates the usage of a sequential Monte Carlo (SMC)
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
from rmc import ConfigDict, LogDensity, SMC

RealArray = ArrayLike

"""
Define energy function
"""
class FunnelE(LogDensity):
    """Evaluate funnel distribution as an energy for SMC sampling.
    """
    def __init__(self, dim: int, x0_sigma: float):#, tempering_fn: Optional[Callable] = None):
        # Store problem definition
        self.dim = dim
        self.x0_sigma = x0_sigma
        self.x0_constant = -0.5 * jnp.log(2 * jnp.pi * x0_sigma**2)
        self.xi_constant = -(dim - 1) * 0.5 * jnp.log(2 * jnp.pi)
        
    def log_likelihood(self, x: RealArray) -> RealArray:
        ll = self.x0_constant - 0.5 * x[..., :1]**2 / self.x0_sigma**2 + self.xi_constant - 0.5 * (self.dim - 1) * x[..., :1]  - 0.5 * jnp.sum(x[..., 1:]**2 / jnp.exp(x[..., :1]), axis=-1, keepdims=True)
        return ll.squeeze()


"""
Configure sampling run.
"""
d = 10              # Dimension of funnel distribution
x0_sigma = 3.       # Value of funnel parameter
# sampling configuration
N = 2000     # Number of particles
T = 256     # Number of tempering scales

# define energy function
Ecl = FunnelE(d, x0_sigma)#, tempering_fn)

# define tempering
sched = jnp.linspace(0, 1, T + 1)
tempering_fn = lambda tstep : sched[tstep]
# cosine
#sched = (1.0 - jnp.cos(jnp.pi * jnp.linspace(0, 1, T + 1))) / 2.
#tempering_fn = lambda tstep : sched[tstep]

# define prior (for initial sampling)
prior_mean = 0.0   # prior mean
prior_std = 1.0    # prior standard deviation


# sampling configuration
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": T,
    "numsteps": 10,
    "numleapfrog": 20,
    "log_freq": 2,
    "energy_cl": Ecl,
    "ESS_thres": 0.98,
    "step_size": 0.02,
    "tempering_fn": tempering_fn
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
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
fig,ax = plt.subplots(1, 1, figsize=(9,5))
ax.scatter(samples[-1, :, 0], samples[-1, :, 1], s=5, marker = 'o', label = 'SMC samples', zorder=0)
ax.axis('equal')
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.legend(loc=2,frameon=False)
plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of SMC for Ionosphere Data
==================================

This script demonstrates the usage of a sequential Monte Carlo (SMC)
sampler for the ionosphere data.
"""


import os
import sys

import jax.numpy as jnp
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import LogisticRegDensity
from utils.io_examples import load_data

from rmc import SMC, ConfigDict

RealArray = ArrayLike

"""
Read ionosphere data and pre-process.
"""
x, y = load_data("examples/datasets/ionosphere_full.pkl")
print(f"Data read shapes: x --> {x.shape}, y --> {y.shape}")
print(f"Range x: min --> {x.min()}, max --> {x.max()}")
print(f"Range y: min --> {y.min()}, max --> {y.max()}")
d = x.shape[1]  # Dimension of features
X = jnp.array(x)
Y = jnp.array(y)

"""
Define density function: logistic regression with ionosphere data read.
"""
# define prior
prior_mean = 0.0  # prior mean on linear regression coeficients
prior_std = 1.0  # prior standard deviation on coeficients
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

noise_std = 1.0
Dcl = LogisticRegDensity(
    d,
    X,
    Y,
    noise_std,
    prior_mean_vec,
    prior_std_vec,
)

"""
Configure sampling run.
"""
# define sampling configuration
N = 2000  # Number of particles
T = 256  # Number of tempering scales

# define tempering
sched = jnp.linspace(0, 1, T + 1)
tempering_fn = lambda tstep: sched[tstep]

# define configuration dictionary
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

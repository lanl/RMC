#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling for linear regression function
=======================================

This example demonstrates the configuration and sampling with different methods
for a linear regression problem.
"""

import sys, os
sys.path.insert(0, os.path.expanduser("/Users/cgarciac/repos/LANL/sampling/rmc-codebase/rmc"))

import jax
import jax.numpy as jnp

from jax.random import multivariate_normal

from rmc import ConfigDict, HMC, LinearRegressionE

"""
Generate data.
"""
D = 200             # Number of (X, Y) pairs
d = 2               # Dimension of x/inputs
# Parameters of linear model
true_w = jnp.linspace(0.3, 0.4, d)
noise_std = 0.1   # fixed noise standard deviation

# random generation
key = jax.random.PRNGKey(0x1234)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)
X = jnp.concatenate((jnp.ones((D, 1)),
        jax.random.normal(x_key, shape=(D, d-1))), 1)
Y = X @ true_w + noise_std * jax.random.normal(call_key, shape=(D,))
print(f"Data generated --> X shape: {X.shape}, Y shape: {Y.shape}")

# define prior
prior_mean = 0.0    # prior mean on linear regression coeficients
prior_std = 10.0    # prior standard deviation on coeficients
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

"""
Configure sampling run.
"""
# define energy function
Ecl = LinearRegressionE(d, X, Y, noise_std, prior_mean_vec, prior_std_vec)

# sampling configuration
N = 30     # Number of samples
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": 10 * N,
    "numsteps": 20,
    "log_freq": 2,
    "energy_cl": Ecl,
    "step_size": 0.01,
}
print(f"Sampling configured --> parameters: {smp_conf}")

"""
Construct sampling object.
"""
hmc_obj = HMC(N, smp_conf)
print("HMC object constructed")

"""
Run sampler.
"""
hmc_obj.sample()

samples = jnp.array(hmc_obj.qall)
print("Collected HMC samples: ", samples.shape)

"""
Plot all samples and true values of linear regression parameters.
"""
import numpy as np
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig,ax = plt.subplots(1, 1, figsize=(9,5))
ax.scatter(samples[:, 0], samples[:, 1], s=5, marker = 'o', color = colors[8], label = 'HMC samples', zorder=0)
ax.scatter(true_w[0], true_w[1], s=25, marker = '*', color = 'k', label = 'True w')
ax.axis('equal')
ax.set_xlabel('x')
ax.set_xlabel('y')
ax.legend(loc=2,frameon=False)
plt.show()

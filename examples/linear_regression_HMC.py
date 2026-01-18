#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sampling for linear regression function
=======================================

This example demonstrates the configuration and sampling with different methods
for a linear regression problem.
"""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rmc import HMC, ConfigDict, LinearRegressionDensity, plot_samples

"""
Generate data.
"""
D = 200  # Number of (X, Y) pairs
d = 2  # Dimension of x/inputs
# Parameters of linear model
true_w = jnp.linspace(0.3, 0.4, d)
noise_std = 0.1  # fixed noise standard deviation

# Random generation
key = jax.random.PRNGKey(0x1234)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)
X = jnp.concatenate((jnp.ones((D, 1)), jax.random.normal(x_key, shape=(D, d - 1))), 1)
Y = X @ true_w + noise_std * jax.random.normal(call_key, shape=(D,))
print(f"Data generated --> X shape: {X.shape}, Y shape: {Y.shape}")

# Define prior
prior_mean = 0.0  # prior mean on linear regression coeficients
prior_std = 10.0  # prior standard deviation on coeficients
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

"""
Configure sampling run.
"""
# Define energy function
Dcl = LinearRegressionDensity(d, X, Y, noise_std, prior_mean_vec, prior_std_vec)

# Sampling configuration
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "maxiter": 300,
    "numleapfrog": 20,
    "log_freq": 2,
    "density_cl": Dcl,
    "step_size": 0.01,
}
print(f"Sampling configured --> parameters: {smp_conf}")

"""
Construct sampling object.
"""
hmc_obj = HMC(smp_conf)
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
from matplotlib import cm
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax = plot_samples(samples, ax, label="HMC samples", size=5, alpha=1, color=colors[8])
ax = plot_samples(
    jnp.atleast_2d(true_w), ax, label="True w", size=30, marker="*", alpha=1, color="k"
)
plt.show()

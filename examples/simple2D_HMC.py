#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Several Examples of 2D Normal Distributions
===========================================

This script includes several 2D normal distribution problems to
demonstrate HMC sampling.
"""


import os
import sys

import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import Norm2D

from rmc import HMC, ConfigDict, plot_samples, plot_trajectories

RealArray = ArrayLike

"""
Set distribution: simple 2D Normal. Use "iso" for isotropic example.
Otherwise, a lopsided distribution will be sampled.
"""
d = 2  # Dimension of x/inputs
example_type = "iso"
if example_type == "iso":
    cov = jnp.eye(d)  # isotropic 2D Gaussian
else:
    cov = jnp.array([[0.2, 0.0], [0.0, 1.0]])  # lopsided Gaussian
rotationAngle = 7 * jnp.pi / 16
R = jnp.array(
    [
        [jnp.cos(rotationAngle), -jnp.sin(rotationAngle)],
        [jnp.sin(rotationAngle), jnp.cos(rotationAngle)],
    ]
)
cov = R.dot(cov).dot(R.T).reshape((1, d, d))

Dcl = Norm2D(cov)

"""
Configure sampling run.
"""
# Random generation
key = jax.random.PRNGKey(3)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)

# Sampling configuration
prior_mean = 0.1
prior_std = 0.5
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,))) ** 2).reshape(
        (1, d, d)
    ),
    "maxiter": 150,
    "numleapfrog": 200,
    "log_freq": 1,
    "density_cl": Dcl,
    "step_size": 0.01,
    "store_path": True,
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

"""
Plot all samples, HMC samples and corresponding trajectories.
"""
qpath = jnp.array(hmc_obj.qpath)
print("Collected samples: ", qpath.shape)

from scipy.stats import multivariate_normal

samples = multivariate_normal(mean=[0.0, 0.0], cov=cov.squeeze()).rvs(size=1000)

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
# Plot base comparison generated with scipy
ax = plot_samples(samples, ax, label="Scipy samples", size=4, alpha=1, color=colors[8])
# Plot HMC results
for i in range(smp_conf["maxiter"]):
    ax = plot_trajectories(
        qpath[i].squeeze(), ax, label="HMC trajectory" if i == 0 else None, zorder=1, color="k"
    )
    ax = plot_samples(
        qpath[i][-1],
        ax,
        label="HMC samples" if i == 0 else None,
        size=30,
        alpha=1,
        zorder=2,
        edgecolor=[],
        facecolor=colors[0],
    )
# Display plots
plt.show()

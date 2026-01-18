#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of 2D Skeleton Density
==============================

This script demonstrates the usage of HMC class for sampling
from a 2D skeleton.
"""


import os
import sys
from functools import partial

import jax.numpy as jnp
from jax.random import multivariate_normal
from jax.typing import ArrayLike

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import Skeleton2D

from rmc import HMC, ConfigDict, plot_samples, plot_trajectories

RealArray = ArrayLike

"""
Set distribution: simple 2D skeleton. This skeleton corresponds to
a slightly rotated O.
"""
D = 100  # Number of points in skeleton
theta = jnp.linspace(0, 2 * jnp.pi, D + 1)[:-1]
z = jnp.vstack((1.0 * jnp.cos(theta), 0.5 * jnp.sin(theta)))
d = z.shape[0]
rotationAngle = 7 * jnp.pi / 16
R = jnp.array(
    [
        [jnp.cos(rotationAngle), -jnp.sin(rotationAngle)],
        [jnp.sin(rotationAngle), jnp.cos(rotationAngle)],
    ]
)
z = R.dot(z).T.reshape((D, d))

sigma = 0.1  # Standard deviation to define the thickness of the skeleton
Dcl = Skeleton2D(z, sigma)

"""
Configure sampling run.
"""
# Create initial distribution
mean0 = 0.1
std0 = 0.5
mean0_arr = mean0 * jnp.ones((1, d))
cov0_arr = jnp.diagflat((std0**2 * jnp.ones((d,)))).reshape((1, d, d))
initialdist = partial(multivariate_normal, mean=mean0_arr, cov=cov0_arr)

# Sampling configuration
prior_mean = 0.1
prior_std = 0.5
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "initial_sampler_cl": initialdist,
    "maxiter": 20,
    "numleapfrog": 200,
    "log_freq": 1,
    "density_cl": Dcl,
    "step_size": 0.02,
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

import numpy as np
from scipy.stats import multivariate_normal as scipymvn

samples = []
for i in range(2000):
    ind = np.random.choice(D, size=1)
    m = z[ind, :].squeeze()
    samples.append(scipymvn(mean=m, cov=np.eye(d) * sigma**2).rvs(size=1))
samples = np.array(samples)


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

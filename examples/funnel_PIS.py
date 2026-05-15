#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of PIS for Funnel Distribution
======================================

This script demonstrates the usage of a Path Integral Sampler (PIS)
for sampling from the funnel distribution.
"""


import os
import sys

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpy as np
from flax import nnx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import FunnelDensity

from rmc import (
    PathIntegralSampler,
    plot_func_xDim_contours,
    plot_quiver,
    plot_samples,
    plot_trajectories,
    save_plot,
)
from rmc.flax.nn_config_dict import NNConfigDict

RealArray = ArrayLike

"""
Set distribution: 10D funnel distribution.
"""
d = 10  # Dimension of funnel distribution
x0_stddev = 3.0  # Standard deviation for component 0

# Define initial distribution
mean0 = 0.0  # initial mean
# 0.815 ok, 0.8175 gone, 0.8165 gone, 0.816 gone, 0.8155 gone, 0.8151 ok, 0.8153 ok,
# 0.8154 gone, 0.81535 gone, 0.81533 mostly ok but few gone
std0 = 0.81532  # initial standard deviation
# Initial mean in array form
mean0_arr = jnp.array(
    [[-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
)  # mean0 * jnp.ones((1, d))
# Initial covariance matrix
cov0_arr = jnp.diagflat(std0**2 * jnp.ones((d,))).reshape((1, d, d))
print(f"Initial --> mean0: {mean0_arr}, cov0: {cov0_arr}")

Dcl = FunnelDensity(d, x0_stddev, mean0_arr, cov0_arr)

"""
Construct Path Integral Sampler (PIS) Model, a Flax neural network (NN) model,
specifically a multi-layer perceptron (MLP).
"""
# NN configuration
layer_widths = [64, 64, 64]  # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "batch_size": 1000,
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "time_embed": True,
    "opt_type": "ADAM",
    "base_lr": 5e-4,
    "max_epochs": 1000,
    "dt_max": 2e-1,
    "max_samples": 12000,
    "nsamples": 4000,
    "eval_every": 100,
    "warm_start": False,
    "max_loss": -5e2,
    "max_subiter": 1,  # 4, #10,
    "has_aux": False,
    "root_path": "./results_pis_funnel/",
}
print(f"Path integral sampling configured --> parameters: {nn_conf}")

"""
Build PIS model.
"""
h = 0.005
T = 40
print(f"PIS parameters --> h: {h}, T: {T}")
PISmodel = PathIntegralSampler(nn_conf, Dcl, h, T, verbose=True)
print("PIS model constructed")

"""
Train model.
"""
PISmodel.train()
print("PIS model trained")

"""
Run sampler.
"""
key = jax.random.PRNGKey(nn_conf["seed"])
key, subkey = jax.random.split(key)
particle_path_, w = PISmodel.sample(nn_conf["nsamples"], subkey)
particles = jnp.array(particle_path_)
print(f"particles shape: {particles.shape}")
print(f"Evolution of {particles.shape[1]} particles during {particles.shape[0]} time steps")

"""
Plot samples for last T step projected into (x0, x1) plane.
"""
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.size": 16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11, 11))

# Plot samples from initial density function
ax1 = plot_samples(particles[0], ax1, size=5, label="Initial samples", color=colors[8])


# Plot samples generated from LFIS method
ax2 = plot_samples(particles[-1], ax2, size=5, label="PIS samples")

# Plot complete time trajectories for 10 samples
for i in range(10):
    ax3 = plot_trajectories(
        particles[:, i, :],
        ax3,
        label=None,
        color="k",
    )
    ax3 = plot_samples(particles[:, i, :], ax3, label=f"s{i+1}" if i < 3 else None)


# Plot initial sample and segment time trajectories for 20 samples
for i in range(20):
    ax4 = plot_trajectories(
        particles[:, i, :],
        ax4,
        label="trajectory" if i == 0 else None,
        zorder=1,
        color="k",
    )
    ax4 = plot_samples(
        jnp.atleast_2d(particles[0, i, :]),
        ax4,
        label="Initial samples" if i == 0 else None,
        size=10,
        zorder=2,
        edgecolor=[],
        facecolor=colors[8],
    )

# Plot samples generated from PIS method
# with overimposed segment trajectories for 10 samples
# and initial position marked as initial samples
for i in range(10):
    ax5 = plot_trajectories(
        particles[:, i, :],
        ax5,
        label="trajectory" if i == 0 else None,
        zorder=1,
        color="k",
    )
    ax5 = plot_samples(
        jnp.atleast_2d(particles[0, i, :]),
        ax5,
        size=30,
        label="Initial samples" if i == 0 else None,
        zorder=2,
        edgecolor=[],
        facecolor=colors[8],
    )
ax5 = plot_samples(
    particles[-1], ax5, size=20, label="PIS samples", edgecolor=[], facecolor=colors[0]
)

# Plot model control
control = PISmodel.nnmodel(particles[-1], h * T)
ax6 = plot_quiver(particles[-1], control, ax6)

# Save plot
save_plot(fig, nn_conf["root_path"] + "samplePIS_funnel.png")

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
# Overlay LFIS results
ax = plot_samples(
    particles[-1],
    ax,
    label="PIS Samples",
    size=5,
    alpha=0.2,
    zorder=1,
    color="red",
)
# Save plot
save_plot(fig, nn_conf["root_path"] + "samplePIS_funnel_overlay.png")

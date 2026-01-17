#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of LFIS for Funnel Distribution
=======================================

This script demonstrates the usage of a Liouville Flow Importance
Sampler for sampling from the funnel distribution.
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
    CosineSchedule,
    LiouvilleFlow,
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
xg0_mean = 0.0  # Mean for components > 0
xg0_stddev = 1.0  # Standard deviation for components > 0
xg0_mean_vec = xg0_mean * jnp.ones((1, d))
xg0_stddev_vec = xg0_stddev * jnp.ones((1, d))

Dcl = FunnelDensity(d, x0_stddev, xg0_mean_vec, xg0_stddev_vec)

"""
Construct Louville Flow (LF) Model, a Flax neural network (NN) model,
specifically a multi-layer perceptron (MLP).
"""
# define prior
prior_mean = 0.0  # prior mean
prior_std = 1.0  # prior standard deviation

# NN configuration
layer_widths = [64, 64, 64]  # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "task": "train",
    "batch_size": 1000,  # 100, #500,#20000,
    "method": "withoutweight",
    # "method": "withweight_resample",
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "opt_type": "ADAM",
    "base_lr": 0.01,  # 0.01, #0.004,
    "max_epochs": 1000,  # 20, #50, #10,#40, #300,
    "mu0_mean": prior_mean * jnp.ones((1, d)),
    "mu0_covariance": jnp.diagflat((prior_std * jnp.ones((d,))) ** 2).reshape((1, d, d)),
    "dt_max": 2e-1,  # 4e-2, #2.5e-2, #0.01,
    "max_samples": 20000,  # 1000,
    "nsamples": 20000,  # 100, #1000, #500,#250,
    "eval_every": 50,
    "warm_start": False,  # True,
    "max_loss": 5e-2,
    "max_subiter": 1,  # 4, #10,
    "has_aux": True,
    "root_path": "./results_funnel/",
}
print(f"Flow-based sampling configured --> parameters: {nn_conf}")
# dt_max 4e-3 is 250 time steps!
"""
Build LF model.
"""
schedule = CosineSchedule()
LFmodel = LiouvilleFlow(nn_conf, Dcl, schedule, verbose=True)
print("LF model constructed")

"""
Train model.
"""
if nn_conf["task"] == "train":
    LFmodel.train()
print("LF model evolution trained")

"""
Run sampler.
"""
key = jax.random.PRNGKey(nn_conf["seed"])
key, subkey = jax.random.split(key)
withw = True
# withw = False
# if nn_conf["method"] == "withwieight" or nn_conf["method"] == "withwieight_resample":
#    withw = True
particle_path_, logw, logz = LFmodel.sample(nn_conf["nsamples"], withw, subkey)
particles = jnp.array(particle_path_)
print(f"Mean log weight: {logw.mean()}")
print(f"logz: {logz}")
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
ax2 = plot_samples(particles[-1], ax2, size=5, label="LF samples")

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

# Plot samples generated from LFIS method
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
    particles[-1], ax5, size=20, label="LF samples", edgecolor=[], facecolor=colors[0]
)

# Plot model velocity
velocity = LFmodel.LFnn(particles[-1])
ax6 = plot_quiver(particles[-1], velocity, ax6)

# Save plot
save_plot(fig, nn_conf["root_path"] + "sampleLF_funnel.png")

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
    label="LF Samples",
    size=5,
    alpha=0.2,
    zorder=1,
    color="red",
)
# Save plot
save_plot(fig, nn_conf["root_path"] + "sampleLF_funnel_overlay.png")

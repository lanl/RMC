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
from utils.density_examples import Norm2D

from rmc import (
    CosineSchedule,
    LiouvilleFlow,
    plot_quiver,
    plot_samples,
    plot_trajectories,
    save_plot,
)
from rmc.flax.nn_config_dict import NNConfigDict

RealArray = ArrayLike

"""
Set distribution: simple 2D Normal. Use "iso" for isotropic example.
Otherwise, a lopsided distribution will be sampled.
"""
d = 2  # Dimension of x/inputs
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
# define prior
prior_mean = 0.0  # prior mean
prior_std = 2.0  # prior standard deviation
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

"""
Construct Louville Flow (LF) Model, a Flax neural network (NN) model,
specifically a multi-layer perceptron (MLP).
"""
# NN configuration
# layer_widths = [64, 64, 64] # number of neurons per layer
layer_widths = [16, 16, 16]  # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "task": "train",
    "batch_size": 200,  # 500,#20000,
    "method": "withoutweight",  # "withweight_resample"
    # "method": "withweight_resample",
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "opt_type": "ADAM",
    "base_lr": 1e-2,
    "max_epochs": 500,
    "dt_max": 2e-1,  # 4e-3,
    "max_samples": 1000,
    "nsamples": 1000,  # 1000,#500,#250,
    "eval_every": 100,
    "warm_start": False,  # True,
    "max_loss": 5e-1,
    "max_subiter": 4,  # 2, #11, #1, #10,
    "has_aux": True,
    "root_path": "./results_s2D/",
}
print(f"Flow-based sampling configured --> parameters: {nn_conf}")

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
print(f"Evolution of {particles.shape[1]} particles during {particles.shape[0]-1} time steps")

"""
Plot samples and trajectories.
"""
from matplotlib import cm
from matplotlib import pyplot as plt

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
save_plot(fig, nn_conf["root_path"] + "sampleLF_s2D.png")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of DDS for 2D Gaussian Mixture
======================================

This script demonstrates the usage of a Denoising Diffusion Sampler (DDS)
for sampling from a nine mode 2D Gaussian Mixture.
"""


import os
import sys

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpy as np
from flax import nnx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.density_examples import NormMix2D

from rmc import (
    plot_quiver,
    plot_samples,
    plot_trajectories,
    save_plot,
)
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.modules.dds import DenoisingDiffusionSampler
from rmc.utils.schedule_diffusion import cosine_beta_schedule

RealArray = ArrayLike

"""
Set distribution: mixture of 2D normal distributions.
"""
d = 2  # Dimension of x/inputs
grid = 1
means = jnp.array(
    [[0 + grid * i, 0 + grid * j] for i in range(-1, 2) for j in range(-1, 2)]
)  # Gaussian centers
sigma2 = 0.012  # Gaussian variance
weights = jnp.ones(means.shape[0])

Dcl = NormMix2D(means, sigma2, weights)

"""
Construct Denoising Diffusion Sampler (DDS) Model. It uses a Flax neural network (NN)
model, specifically a multi-layer perceptron (MLP).
"""
# NN configuration
layer_widths = [64, 64, 64]  # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 0,
    "batch_size": 500,
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "time_embed": False,
    "opt_type": "ADAM",
    "base_lr": 1e-2,
    "max_epochs": 2000,
    "dt_max": 4e-2,
    "max_samples": 5000,
    "nsamples": 5000,
    "eval_every": 100,
    "max_loss": -5e1,
    "max_subiter": 1,
    "has_aux": False,
    "root_path": "./results_dds_mix2D/",
}
print(f"Denoising diffusion sampler configured --> parameters: {nn_conf}")

"""
Build DDS model.
"""
sigma = 0.05
K = 25
beta_schedule = cosine_beta_schedule
print(f"DDS parameters --> sigma: {sigma}, K: {K}, beta-schedule: {beta_schedule}")
DDSmodel = DenoisingDiffusionSampler(nn_conf, Dcl, sigma, K, beta_schedule, verbose=True)
print("DDS model constructed")

"""
Train model.
"""
DDSmodel.train()
print("DDS model trained")

"""
Run sampler.
"""
key = jax.random.PRNGKey(nn_conf["seed"])
key, subkey = jax.random.split(key)
particle_path_ = DDSmodel.sample(nn_conf["nsamples"], subkey)
particles = jnp.array(particle_path_)
print(f"particles shape: {particles.shape}")
print(f"Evolution of {particles.shape[1]} particles during {particles.shape[0]} time steps")


"""
Plot samples for last T step projected into (x0, x1) plane.
"""
from matplotlib import cm
from matplotlib import pyplot as plt

plt.rcParams.update({"font.size": 16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11, 11))

# Plot samples from initial density function
ax1 = plot_samples(particles[0], ax1, size=5, label="Initial samples", color=colors[8])

# Plot samples generated from LFIS method
ax2 = plot_samples(particles[-1], ax2, size=5, label="DDS samples")

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

# Plot samples generated from DDS method
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
    particles[-1], ax5, size=20, label="DDS samples", edgecolor=[], facecolor=colors[0]
)

# Plot learned model (pseudo-score)
pseudo_score = DDSmodel.nnmodel(particles[-1], K)
ax6 = plot_quiver(particles[-1], pseudo_score, ax6)

# Save plot
save_plot(fig, nn_conf["root_path"] + "sampleDDS_Mix2D.png")

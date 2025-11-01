#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of LFIS for Funnel Distribution
=======================================

This script demonstrates the usage of a Liouville Flow Importance
Sampler for sampling from the funnel distribution.
"""

from typing import Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jax.random import multivariate_normal

from flax import nnx

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rmc import ConfigDict, LogDensity
from rmc.flax.liouville_flow import LiouvilleFlow
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.schedule import CosineSchedule

from distributions_examples import ENorm2D

RealArray = ArrayLike

"""
Set distribution: simple 2D Normal. Use "iso" for isotropic example. 
Otherwise, a lopsided distribution will be sampled.
"""
d = 2   # Dimension of x/inputs
example_type = "iso"
if example_type == "iso":
    cov = jnp.eye(d)  # isotropic 2D Gaussian
else:
    cov = jnp.array([[0.2, 0.0], [0.0, 1.0]])  # lopsided Gaussian
rotationAngle = 7 * jnp.pi / 16
R = jnp.array([[jnp.cos(rotationAngle), -jnp.sin(rotationAngle)], [jnp.sin(rotationAngle), jnp.cos(rotationAngle)]])
cov = R.dot(cov).dot(R.T).reshape((1, d, d))

Ecl = ENorm2D(cov)

"""
Configure sampling run.
"""
# define prior
prior_mean = 0.0   # prior mean
prior_std = 1.0    # prior standard deviation
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

"""
Construct Louville Flow (LF) Model, a Flax neural network (NN) model, 
specifically a multi-layer perceptron (MLP). 
"""
# NN configuration
layer_widths = [64, 64] # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "task": "train",
    "batch_size": 100, #500,#20000,
    "method": "withoutweight", #"withweight_resample"
    #"method": "withweight_resample",
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "opt_type": "ADAM",
    "base_lr": 0.01,
    "max_epochs": 50, #20, #40, #50, #10, #300,
    "mu0_mean": prior_mean * jnp.ones((1, d)),
    "mu0_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "dt_max": 5e-2, #2e-1, #2.5e-2, #5e-2, #0.01,
    "max_samples": 1000,
    "nsamples": 500, #1000,#500,#250,
    "eval_every": 1,
    "warm_start": True,
    "max_loss": 2e-1,
    "max_subiter": 1, #10,
}
print(f"Flow-based sampling configured --> parameters: {nn_conf}")

"""
Build LF model.
"""
schedule = CosineSchedule()
LFmodel = LiouvilleFlow(nn_conf, Ecl, schedule, verbose=True)
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
particle_path_, w, logz = LFmodel.sample()
particles = jnp.array(particle_path_)
print(f"Mean weight: {w.mean()}")
print(f"logz: {logz}")
print(f"particles shape: {particles.shape}")
print(f"Evolution of {particles.shape[1]} particles during {particles.shape[0]} time steps")

"""
Plot samples for last T step projected into (x0, x1) plane.
"""
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11,11))
ax1.scatter(particles[0, :, 0], particles[0, :, 1], s=5, marker = 'o', label = 'Initial samples', zorder=0)
ax1.axis('equal')
ax1.set_xlabel('x0')
ax1.set_ylabel('x1')
ax1.legend(loc=2,frameon=False)

ax2.scatter(particles[-1, :, 0], particles[-1, :, 1], s=5, marker = 'o', label = 'LF samples', zorder=0)
ax2.axis('equal')
ax2.set_xlabel('x0')
ax2.set_ylabel('x1')
ax2.legend(loc=2,frameon=False)

ax3.axis('equal')
for i in range(20):
    ax3.plot(particles[:,i,0], particles[:,i,1], color = 'k', linestyle=':', label='trajectory' if i==0 else None, zorder = 1)
    ax3.scatter(particles[0,i,0], particles[0,i,1], edgecolor=[], facecolor = colors[8], label='samples0' if i==0 else None, zorder = 2)

ax3.set_xlabel('x0')
ax3.set_ylabel('x1')
ax3.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))

ax4.axis('equal')
for i in range(10):
    ax4.plot(particles[:,i,0], particles[:,i,1], color = 'k', linestyle=':', label='trajectory' if i==0 else None, zorder = 1)
    ax4.scatter(particles[0,i,0], particles[0,i,1], edgecolor=[], facecolor = colors[8], label='samples0' if i==0 else None, zorder = 2)

ax4.scatter(particles[-1,:,0], particles[-1,:,1], edgecolor=[], facecolor = colors[0], label='samplesF')
ax4.set_xlabel('x0')
ax4.set_ylabel('x1')
ax4.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))
plt.show()

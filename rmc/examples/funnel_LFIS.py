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
from rmc import ConfigDict
from rmc.modules.lfis import LiouvilleFlow
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.utils.schedule import CosineSchedule

from utils.distributions_examples import FunnelE

RealArray = ArrayLike

"""
Set distribution: 10D funnel distribution.
"""
d = 10                  # Dimension of funnel distribution
x0_stddev = 3.          # Standard deviation for component 0
xg0_mean = 0.0          # Mean for components > 0
xg0_stddev = 1.0        # Standard deviation for components > 0
xg0_mean_vec = xg0_mean * jnp.ones((1, d))
xg0_stddev_vec = xg0_stddev * jnp.ones((1, d))

Ecl = FunnelE(d, x0_stddev, xg0_mean_vec, xg0_stddev_vec)

"""
Construct Louville Flow (LF) Model, a Flax neural network (NN) model, 
specifically a multi-layer perceptron (MLP). 
"""
# define prior
prior_mean = 0.0   # prior mean
prior_std = 1.0    # prior standard deviation

# NN configuration
layer_widths = [64, 64, 64] # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "task": "train",
    "batch_size": 1000, #100, #500,#20000,
    "method": "withoutweight",
    #"method": "withweight_resample",
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "opt_type": "ADAM",
    "base_lr": 0.01, #0.01, #0.004,
    "max_epochs": 1000, #20, #50, #10,#40, #300,
    "mu0_mean": prior_mean * jnp.ones((1, d)),
    "mu0_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "dt_max": 2e-1, #4e-2, #2.5e-2, #0.01,
    "max_samples": 20000, #1000,
    "nsamples": 20000, #100, #1000, #500,#250,
    "eval_every": 50,
    "warm_start": False, #True,
    "max_loss": 5e-2,
    "max_subiter": 1, #4, #10,
    "has_aux": True,
    "root_path": "/Users/cgarciac/repos/LANL/sampling/RMC/rmc/results_funnel/"
}
print(f"Flow-based sampling configured --> parameters: {nn_conf}")
# dt_max 4e-3 is 250 time steps!
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
#LFmodel.tlst = [0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48]

"""
Run sampler.
"""
key = jax.random.PRNGKey(nn_conf["seed"])
key, subkey = jax.random.split(key)
withw = True
#withw = False
#if nn_conf["method"] == "withwieight" or nn_conf["method"] == "withwieight_resample":
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
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(11,11))
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

for i in range(10):
    ax3.plot(particles[:,i,0], particles[:,i,1], color = 'k', linestyle=':', label='trajectory')
    ax3.scatter(particles[:,i,0], particles[:,i,1])
ax3.axis('equal')
ax3.set_xlabel('x0')
ax3.set_ylabel('x1')

ax4.axis('equal')
for i in range(100):
    ax4.plot(particles[:,i,0], particles[:,i,1], color = 'k', linestyle=':', label='trajectory' if i==0 else None, zorder = 1)
    ax4.scatter(particles[0,i,0], particles[0,i,1], edgecolor=[], facecolor = colors[8], label='samples0' if i==0 else None, zorder = 2)

ax4.set_xlabel('x0')
ax4.set_ylabel('x1')
ax4.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))

scale = 10

ax5.axis('equal')
#for i in range(10):
#    ax5.plot(particles[:,i,0], particles[:,i,1], color = 'k', linestyle=':', label='trajectory' if i==0 else None, zorder = 1)
#    ax5.scatter(particles[0,i,0], particles[0,i,1], edgecolor=[], facecolor = colors[8], label='samples0' if i==0 else None, zorder = 2)
ax5.quiver(particles[0,:,0], particles[0,:,1], particles[1,:,0], particles[1,:,1], angles='xy', scale_units='xy', scale=scale, )
#ax5.scatter(particles[-1,:,0], particles[-1,:,1], edgecolor=[], facecolor = colors[0], label='samplesF')
ax5.set_xlabel('x0')
ax5.set_ylabel('x1')
#ax5.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))

ax6.axis('equal')
ax6.quiver(particles[0,:,0], particles[0,:,1], particles[-1,:,0], particles[-1,:,1], angles='xy', scale_units='xy', scale=scale, )
ax6.set_xlabel('x0')
ax6.set_ylabel('x1')
#plt.show()
plt.savefig(nn_conf["root_path"] + "sampleLF_funnel.png")

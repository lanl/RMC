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
from rmc import ConfigDict, LogDensityPath
from rmc.flax.liouville_flow import LiouvilleFlow
from rmc.flax.nn_config_dict import NNConfigDict
from rmc.flax.schedule import CosineSchedule

RealArray = ArrayLike

"""
Define energy function
"""
class FunnelE(LogDensityPath):
    """Evaluate funnel distribution as an energy for SMC sampling.
    """
    def __init__(self, dim: int, x0_sigma: float, mean_prior: ArrayLike, stddev_prior: ArrayLike,):
        # Store problem definition
        self.dim = dim
        self.x0_sigma = x0_sigma
        self.x0_constant = -0.5 * jnp.log(2 * jnp.pi * x0_sigma**2)
        self.xi_constant = -(dim - 1) * 0.5 * jnp.log(2 * jnp.pi)
        # Store parameters of base distribution
        self.mean_prior = jnp.array(mean_prior)
        self.precision_prior = jnp.diagflat(1. /  jnp.array(stddev_prior)**2)
        self.log_det_prior = -self.dim / 2. * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(stddev_prior)**2) / 2.
        
    def log_base(self, x: RealArray) -> RealArray:
        xvec = (x - self.mean_prior).reshape((-1, 1, x.shape[-1]))
        lb = self.log_det_prior - 0.5 * jnp.squeeze(xvec @ self.precision_prior @ jnp.transpose(xvec, axes=(0, 2, 1)))
        return lb
        
    def log_target(self, x: RealArray) -> RealArray:
        lt = self.x0_constant - 0.5 * x[..., :1]**2 / self.x0_sigma**2 + self.xi_constant - 0.5 * (self.dim - 1) * x[..., :1]  - 0.5 * jnp.sum(x[..., 1:]**2 / jnp.exp(x[..., :1]), axis=-1, keepdims=True)
        return lt.squeeze()


"""
Configure sampling run.
"""
d = 10              # Dimension of funnel distribution
x0_sigma = 3.       # Value of funnel parameter

# define prior
prior_mean = 0.0   # prior mean
prior_std = 1.0    # prior standard deviation
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

# define energy function
Ecl = FunnelE(d, x0_sigma, prior_mean_vec, prior_std_vec)

"""
Construct Louville Flow (LF) Model, a Flax neural network (NN) model, 
specifically a multi-layer perceptron (MLP). 
"""
# NN configuration
layer_widths = [64, 64, 64] # number of neurons per layer
nn_conf: NNConfigDict = {
    "seed": 10,
    "task": "train",
    "batch_size": 50,#20000,
    "method": "withoutweight",
    "dim": d,
    "layer_widths": layer_widths,
    "activation_func": nnx.silu,
    "opt_type": "ADAM",
    "base_lr": 0.004,
    "max_epochs": 40, #300,
    "mu0_mean": prior_mean * jnp.ones((1, d)),
    "mu0_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "dt_max": 0.01,
    "nsamples": 250,
    "eval_every": 1,
}
print(f"Flow-based sampling configured --> parameters: {nn_conf}")

"""
Build and train LF model.
"""
schedule = CosineSchedule()
LFmodel = LiouvilleFlow(nn_conf, Ecl, schedule, verbose=True)
if nn_conf["task"] == "train":
    LFmodel.train()

"""
Construct sampling object.
"""
#smc_obj = SMC(N, T, smp_conf)
#print("SMC object constructed")

"""
Run sampler.
"""
#smc_obj.sample()

#samples = jnp.array(smc_obj.hmc_.qall)
#print("Collected SMC samples: ", samples.shape)

"""
Plot samples for last T step projected into (x0, x1) plane.
"""
#from matplotlib import pyplot as plt, cm
#plt.rcParams.update({'font.size':16})
#fig,ax = plt.subplots(1, 1, figsize=(9,5))
#ax.scatter(samples[-1, :, 0], samples[-1, :, 1], s=5, marker = 'o', label = 'SMC samples', zorder=0)
#ax.axis('equal')
#ax.set_xlabel('x0')
#ax.set_ylabel('x1')
#ax.legend(loc=2,frameon=False)
#plt.show()

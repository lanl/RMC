#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Several Examples of 2D Normal Distributions
===========================================

This script includes several 2D normal distribution problems to
demonstrate Stein variational gradient descent (SVGD) sampling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jax.random import multivariate_normal

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rmc import ConfigDict, SVGD

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
# random generation
key = jax.random.PRNGKey(3)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)

# sampling configuration
N = 100     # Number of particles

# define prior
prior_mean = 0.01  # prior mean
prior_std = 0.5    # prior standard deviation
prior_mean_vec = prior_mean * jnp.ones((1, d))
prior_std_vec = prior_std * jnp.ones((1, d))

smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (N, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": 150,
    "log_freq": 10,
    "energy_cl": Ecl,
    "step_size": 0.01,
    "kernel_parameter": -1, # Compute median
    "update_weight": 0.9, # Alpha in code
}
print(f"Sampling configured --> parameters: {smp_conf}")

"""
Construct sampling object.
"""
svgd_obj = SVGD(N, smp_conf)
print("SVGD object constructed")

"""
Run sampler.
"""
qall = svgd_obj.sample()
print("Collected SVGD samples: ", qall.shape)

"""
Plot all samples, SVGD samples and corresponding trajectories.
"""

from scipy.stats import multivariate_normal
samples = multivariate_normal(mean=[0.,0.], cov=cov.squeeze()).rvs(size=1000)

import numpy as np
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig,ax = plt.subplots(1, 1, figsize=(9,5))
ax.scatter(samples[:, 0], samples[:, 1], s=1, marker = 'o', color = colors[8], label = 'MC samples', zorder=0)
ax.axis('equal')
ax.scatter(qall[:,0], qall[:,1], edgecolor=[], facecolor = colors[0], label='SVGD samples')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))
plt.show()

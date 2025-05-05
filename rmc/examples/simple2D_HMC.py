#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Several Examples of 2D Normal Distributions
===========================================

This script includes several 2D normal distribution problems to
demonstrate HMC sampling.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from jax.random import multivariate_normal

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rmc import ConfigDict, LogDensity, HMC, LinearRegressionE

RealArray = ArrayLike

"""
Define energy function
"""
class ENorm2D(LogDensity):
    def __init__(self, cov):
        self.cov = cov
        self.invcov = jnp.linalg.inv(cov)

    def log_likelihood(self, x: RealArray) -> RealArray:
        ll = -0.5 * x @ self.invcov @ x.T
        return ll.squeeze()


"""
Define distribution. Use "iso" for isotropic example. Otherwise, a
lopsided distribution will be sampled.
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

"""
Configure sampling run.
"""
# random generation
key = jax.random.PRNGKey(3)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)


Ecl = ENorm2D(cov)

# sampling configuration
prior_mean = 0.1
prior_std = 0.5
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": 150,
    "numsteps": 200,
    "log_freq": 1,
    "energy_cl": Ecl,
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
samples = multivariate_normal(mean=[0.,0.], cov=cov.squeeze()).rvs(size=1000)

import numpy as np
from matplotlib import pyplot as plt, cm
plt.rcParams.update({'font.size':16})
colors = cm.plasma(np.linspace(0, 1, 12))

fig,ax = plt.subplots(1, 1, figsize=(9,5))
ax.scatter(samples[:, 0], samples[:, 1], s=1, marker = 'o', color = colors[8], label = 'MC samples', zorder=0)
ax.axis('equal')

for i in range(smp_conf["maxiter"]):
    ax.plot(qpath[i][:,0,0], qpath[i][:,0,1], color = 'k', linestyle=':', label='HMC trajectory' if i==0 else None, zorder = 1)
    ax.scatter(qpath[i][0,0,0], qpath[i][0,0,1], edgecolor=[], facecolor = colors[0], label='HMC samples' if i==0 else None, zorder = 2)

ax.scatter(qpath[-1][-1,0,0], qpath[-1][-1,0,1], edgecolor=[], facecolor = colors[0])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc=2,frameon=False)#,bbox_to_anchor=(1.0, 0.7))
plt.show()

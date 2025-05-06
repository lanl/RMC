#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of 2D Skeleton Distribution
===================================

This script includes demonstrates the usage of HMC class for sampling
from a 2D skeleton.
"""

from functools import partial
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
class ESkeleton2D(LogDensity):
    def __init__(self, z, sigma):
        self.z = jnp.array(z)
        self.sigma = sigma
        self.numd, self.dim = self.z.shape

    def mvnpdfsum(self, x, i, psum):
        return psum + jax.scipy.stats.multivariate_normal.pdf(x, mean=self.z[i, :], cov=jnp.eye(self.dim) * self.sigma**2)

    def log_likelihood(self, x: RealArray) -> RealArray:
        funcbody = partial(self.mvnpdfsum, jnp.ravel(x))
        p = jax.lax.fori_loop(0, self.numd, funcbody, 0.)
        ll = jnp.log(p) - jnp.log(self.numd)
        return ll.squeeze()


"""
Define distribution.
"""
D = 100             # Number of points in skeleton
theta = jnp.linspace(0, 2 * jnp.pi, D + 1)[:-1]
z = jnp.vstack((1.0 * jnp.cos(theta), 0.5 * jnp.sin(theta)))
d = z.shape[0]
rotationAngle = 7 * jnp.pi / 16
R = jnp.array([[jnp.cos(rotationAngle), -jnp.sin(rotationAngle)], [jnp.sin(rotationAngle), jnp.cos(rotationAngle)]])
z = R.dot(z).T.reshape((D, d))

"""
Configure sampling run.
"""
# random generation
key = jax.random.PRNGKey(3)
key, x_key = jax.random.split(key)
key, call_key = jax.random.split(key)

sigma = 0.1
Ecl = ESkeleton2D(z, sigma)

# sampling configuration
prior_mean = 0.1
prior_std = 0.5
smp_conf: ConfigDict = {
    "seed": 0,
    "sample_shape": (1, d),
    "initial_sampler_fn": multivariate_normal,
    "initial_sampler_mean": prior_mean * jnp.ones((1, d)),
    "initial_sampler_covariance": jnp.diagflat((prior_std * jnp.ones((d,)))**2).reshape((1, d, d)),
    "maxiter": 20,
    "numleapfrog": 200,
    "log_freq": 1,
    "energy_cl": Ecl,
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
    m = z[ind,:].squeeze()
    samples.append(scipymvn(mean=m, cov=np.eye(d)*sigma**2).rvs(size=1))
samples = np.array(samples)


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
ax.legend(loc=2,frameon=False)
plt.show()


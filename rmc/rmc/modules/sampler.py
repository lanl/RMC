# -*- coding: utf-8 -*-

"""Definitions for sampler modules."""

from functools import partial

import jax
import jax.numpy as jnp

from typing import Tuple
from jax.typing import ArrayLike

import copy

from rmc.utils.config_dict import ConfigDict

RealArray = ArrayLike

class Sampler:
    """Base sampler class.

    A :class:`Sampler` is the base class for all the sampling methods implemented.
    """
    def __init__(
        self,
        config: ConfigDict,

    ):
        """
        Args:
            maxiter: Maximum number of iteration in sampling algorithm.
        """
        self.config = config
        self.itnum = 0
        self.maxiter = config["maxiter"]

    def draw_initial_sample(self, key: ArrayLike):
        """Perform the initial sampling."""
        key, subkey = jax.random.split(key)
        #print("mean shape: ", self.config["initial_sampler_mean"].shape)
        #print("cov shape: ", self.config["initial_sampler_covariance"].shape)
        initial_sampler = partial(self.config["initial_sampler_fn"],
                            mean = self.config["initial_sampler_mean"],
                            cov = self.config["initial_sampler_covariance"],)

        return initial_sampler(subkey, shape = (self.config["sample_shape"][0],))

    def post_initialization(self, key: ArrayLike):
        """Perform required random state initialization."""

    def step(self, key: ArrayLike, prev_samples: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Perform a single sampler step."""

    def sample(self):
        """Initialize and run the sampling algorithm."""
        key = jax.random.PRNGKey(self.config["seed"])
        key, subkey = jax.random.split(key)
        samples = self.draw_initial_sample(subkey)
        #print("In base sample, initial sample shape: ", samples.shape)
        key = self.post_initialization(key, samples)

        for self.itnum in range(self.itnum, self.itnum + self.maxiter):
            key, samples = self.step(key, samples)
            if self.itnum % self.config["log_freq"] == 0:
                self.print_stats()
        return samples


def accept_Metropolis(key: ArrayLike, deltaE: float):
    """Apply Metropolis acceptance criterion.

    Args:
        key: Key for random generation.
        deltaE: Change in energy (Enew - Eold).

    Returns:
        `True` if the change in energy is negative (i.e. energy
        decreases) or if the random generation makes the increment in
        energy acceptable. Otherwise, `False` is returned.
    """

    probL = jnp.log(jax.random.uniform(key, shape=(deltaE.shape[0],)))
    acc = probL < -deltaE
    #if probL < -deltaE:
    #    acc = True
    #else:
    #    acc = False

    #print(f"deltaE: {deltaE}, acc: {acc}")
    return acc.reshape((-1, 1))

### Mass matrix for HMC
### Add a energy class

class HMC(Sampler):
    """Hamiltonian Monte Carlo (HMC) sampler."""
    def __init__(self, Nsamples: int, config: ConfigDict):
        """Initialization of HMC class.
        Args:
            Nsamples: Number of samples to draw.
            config: Dictionary with sampler configuration parameters.
        """
        self.Nsamples = Nsamples
        super().__init__(config)
        # State variables
        self.q_ = jnp.zeros(self.config["sample_shape"])
        # Momentum variables (renormalized such that no separate mass is used)
        self.p_rn_ = jnp.zeros(self.config["sample_shape"])
        # State derivative
        self.q_der_ = jnp.zeros(self.config["sample_shape"])
        # Size of the step for updating the Hamiltonian dynamics
        # (independent for each component)
        self.step_size_ = jnp.zeros(self.config["sample_shape"])

        # Energy function
        self.E_cl = self.config["energy_cl"]

        # Initialization
        # Number of current acceptances
        self.acceptances = 0
        # Flag to avoid extra computation of derivative
        # (true just in first dynamic step for first iter)
        self.first_derivative = True

    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform random initialization of required state variables."""
        # Initialize state
        self.q_ = samples.copy()
        # Initialize momentum variables randomly
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.p_rn_ = jax.random.normal(subkey1, shape = self.config["sample_shape"])
        # Initialize step size for updating Hamiltonian dynamics randomly
        self.step_size_ = self.update_stepsize(subkey2)
        return key

    def update_stepsize(self, key: ArrayLike, min_step_size: float = 1e-4,
            delta_step_size: float = 5e-3
        ):
        """Update size of the step.

        This is the step for advancing the Hamiltonian dynamics and is
        independent for each component in the state.

        Args:
            key: Key for jax random generation.
            min_step_size: Minimum step size to allow.
            delta_step_size: Maximum increment to allow for the step size.

        Returns:
            A randomly generated step size for each component of the
            state that is between (min_step_size,
            min_step_size + delta_step_size).
        """
        step_size = min_step_size + jax.random.uniform(key,
                        shape=self.config["sample_shape"]) * delta_step_size
        return step_size

    def step(self, key: ArrayLike, prev_samples: ArrayLike,) -> Tuple[ArrayLike, ArrayLike]:
        """Compute one step of sampler."""
        numsteps = self.config["numsteps"]
        # Store old state
        q_old = self.q_.copy()
        p_rn_old = self.p_rn_.copy()
        self.Eold = self.compute_energy()
        # Advance state via leapfrog discretization
        self.compute_leapfrog_step(numsteps)
        #print("In HMC step, after leapfrog --> q.shape: ", self.q_.shape)
        #print("In HMC step, after leapfrog --> p_rn.shape: ", self.p_rn_.shape)
        # Negate momentum variables p
        self.p_rn_ = -self.p_rn_
        # Compute new energy
        self.Enew = self.compute_energy()
        # Metropolis accept/reject
        key, subkey1, subkey2 = jax.random.split(key, 3)

        accept = accept_Metropolis(subkey1, self.Enew - self.Eold)
        self.q_ = accept * self.q_ + (1 - accept) * q_old
        self.p_rn_ = accept * self.p_rn_ + (1 - accept) * p_rn_old

        self.acceptances = accept.mean()

        #if accept_Metropolis(subkey1, self.Enew - self.Eold):
        #    self.acceptances = self.acceptances + 1
        #else: # Return to previous state
        #    self.q_ = q_old
        #    self.p_rn_ = p_rn_old

        samples = self.q_

        # Update step size for Hamiltonian dynamics
        self.step_size_ = self.update_stepsize(subkey2)

        return key, samples


    def compute_energy(self):
        """Compute energy of system.

        The energy of the system is the sum of potential and kinetic
        energy. The potential energy is. The kinetic energy is half
        the sum of the square of the momemtum of the system.
        """
        potentialE = -self.E_cl.log_unposterior(self.q_) # potential energy
        kineticE = jnp.sum(self.p_rn_**2, axis=1) / 2.0 # kinetic energy
        #print("potentialE.shape: ", potentialE.shape)
        #print("kineticE.shape: ", kineticE.shape)
        return potentialE + kineticE

    def compute_leapfrog_step(self, numsteps: int = 1):
        """Advance state using leapfrog numerical discretization."""
        #print("In leapfrog, q.shape: ", self.q_.shape)
        if self.first_derivative: # First time that derivative is computed
            self.q_der_ = -self.E_cl.der_log_unposterior(self.q_) # update der_q
            self.first_der = False
        # Initial half p-step
        self.p_rn_ = self.p_rn_ - self.step_size_ * self.q_der_ / 2.
        for i in range(numsteps-1):
            # q-step
            self.q_ = self.q_ + self.step_size_ * self.p_rn_
            # Half p-steps merged
            self.q_der_ = -self.E_cl.der_log_unposterior(self.q_) # update der_q_
            self.p_rn_ = self.p_rn_ - self.step_size_ * self.q_der_

            #print(f"{i}: q: {self.q_}, q_der: {self.q_der_}, p: {self.p_rn_}")

        # q-step
        self.q_ = self.q_ + self.step_size_ * self.p_rn_

        # Final half p-step
        self.q_der_ = -self.E_cl.der_log_unposterior(self.q_) # update der_q_
        self.p_rn_ = self.p_rn_ - self.step_size_ * self.q_der_ / 2.

    def print_stats(self):
        """Print statistics computed during sample generation."""
        #print(f"Iter: {self.itnum}, acceptances: {self.acceptances}")
        print(f"Iter: {self.itnum}, acceptances: {self.acceptances}, Eold: {self.Eold}, Enew: {self.Enew}, state: {self.q_}")


class SMC(Sampler):
    """Sequential Monte Carlo sampler."""
    def __init__(self, Nsamples: int, T: int, config: ConfigDict):
        """
        Args:
            Nsamples: Number of samples to draw.
            T: Number of tempering scales.
            config: Dictionary with sampler configuration parameters.
        """
        self.Nsamples = Nsamples
        self.T = T
        super().__init__(config)
        # Unnormalized log-weights
        self.logw = jnp.log(jnp.ones(self.Nsamples) / self.Nsamples)
        # Unnormalized weights
        self.w = jnp.exp(self.logw)
        # Normalized weights
        self.W = self.w / self.w.sum()
        # log of normalization constant
        self.logZ = 0.

        # Sampler state
        self.x = jnp.zeros(self.config["sample_shape"])

        # HMC Object
        self.hmc_ = HMC(Nsamples, config)

        # Energy function
        self.E_cl = self.config["energy_cl"]
        # Effective sample size threshold for resampling
        self.ESS_threshold = self.config["ESS_thres"]


    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform random initialization of required state variables."""
        # Initialize HMC state
        self.hmc_.q_ = samples.copy()
        #print("In SMC, HMC state shape is: ", self.hmc_.q_.shape)
        # Initialize momentum variables randomly
        key, subkey = jax.random.split(key)
        self.hmc_.p_rn_ = jax.random.normal(subkey, shape = self.hmc_.q_.shape)
        # Initialize step size for updating Hamiltonian dynamics
        self.hmc_.step_size_ = 0.02
        return key


    def step(self, key: ArrayLike, prev_samples: ArrayLike,) -> Tuple[ArrayLike, ArrayLike]:
        """Compute one step of sampler."""
        dlogtw = (self.E_cl.log_unposterior(self.x, self.itnum + 1) - self.E_cl.log_unposterior(self.x, self.itnum)).squeeze()
        self.logw = self.logw + dlogtw

        ess = self.compute_ess()
        if ess < self.config["ESS_thres"]:
            self.logZ = self.logZ + jax.scipy.special.logsumexp(self.logw, axis=-1)
            key, subkey = jax.random.split(key)
            self.resample(subkey)
            print(f"!Resampled at tStep={self.itnum}; logZ = {self.logZ}")

        # Advance sample via HMC
        #print("Starting HMC step inside SMC step")
        key, samples = self.hmc_.step(key, prev_samples)

        return key, samples

    def compute_ess(self):
        """Compute effective sample size (ESS)."""
        first_term = 2. * jax.scipy.special.logsumexp(self.logw, axis=-1)
        second_term = jax.scipy.special.logsumexp(2. * self.logw, axis=-1)
        return jnp.exp(first_term - second_term) / self.Nsamples


    def resample(self, key: ArrayLike):
        """Resample particles."""
        index = jnp.random.choice(key, jnp.arange(self.Nsamples), shape=self.Nsamples,
                    replace=True, p = jax.scipy.special.softmax(self.logw, axis=-1),)
        self.x = self.x[index,:]
        self.logw = jnp.log(jnp.ones(self.Nsamples) / self.Nsamples)


    def print_stats(self):
        """Print statistics computed during sample generation."""
        logZ = self.logZ + jax.scipy.special.logsumexp(self.logw, axis=-1)
        print(f"Iter: {self.itnum}, acceptances: {self.hmc_.acceptances}, ESS: {self.compute_ess()}, logZ: {logZ}")

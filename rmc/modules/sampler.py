# -*- coding: utf-8 -*-

"""Definitions for sampler modules."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from rmc.utils.config_dict import ConfigDict

RealArray = ArrayLike


class Sampler:
    """Base sampler class.

    This class is the base class for all the sampling methods implemented.
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

        # Tempering function
        if "tempering_fn" in self.config.keys():
            self.tempering_fn = self.config["tempering_fn"]
            self.tempering = self.tempering_fn(0)
        else:
            self.tempering_fn = None
            self.tempering = None

    def draw_initial_sample(self, key: ArrayLike):
        """Perform the initial sampling."""
        key, subkey = jax.random.split(key)
        initial_sampler = partial(
            self.config["initial_sampler_fn"],
            mean=self.config["initial_sampler_mean"],
            cov=self.config["initial_sampler_covariance"],
        )

        return initial_sampler(subkey, shape=(self.config["sample_shape"][0],))

    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform required random state initialization."""

    def step(self, key: ArrayLike, prev_samples: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Perform a single sampler step."""

    def sample(self):
        """Initialize and run the sampling algorithm."""
        key = jax.random.PRNGKey(self.config["seed"])
        key, subkey = jax.random.split(key)
        samples = self.draw_initial_sample(subkey)
        key = self.post_initialization(key, samples)

        for self.itnum in range(self.itnum, self.itnum + self.maxiter):
            if self.tempering_fn is not None:
                self.tempering = self.tempering_fn(self.itnum)

            key, samples = self.step(key, samples)
            if self.itnum % self.config["log_freq"] == 0:
                self.print_stats()
        self.print_stats()
        return samples


def accept_Metropolis(key: ArrayLike, E_old, E_new):
    """Apply Metropolis acceptance criterion.

    (Use exponential form.)

    Args:
        key: Key for random generation.
        deltaE: Change in energy (Enew - Eold).

    Returns:
        `True` if the change in energy is negative (i.e. energy
        decreases) or if the random generation makes the increment in
        energy acceptable. Otherwise, `False` is returned.
    """
    prob = jax.random.uniform(key, shape=(E_old.shape[0],))
    acc = prob < jnp.exp(E_old - E_new)
    acc = acc.reshape((-1, 1)).astype("float32")
    return acc


### Mass matrix for HMC
class HMC(Sampler):
    """Hamiltonian Monte Carlo (HMC) sampler."""

    def __init__(self, config: ConfigDict):
        """Initialization of HMC class.
        Args:
            config: Dictionary with sampler configuration parameters.
        """
        super().__init__(config)
        # State variables
        self.q_ = jnp.zeros(self.config["sample_shape"])
        # Momentum variables (renormalized such that no separate mass is used)
        self.p_rn_ = jnp.zeros(self.config["sample_shape"])
        # State derivative
        self.q_der_ = jnp.zeros(self.config["sample_shape"])
        # Size of the step for updating the Hamiltonian dynamics
        if "step_size" in self.config.keys():  # Fix value given
            self.step_size_ = self.config["step_size"] * jnp.ones(self.config["sample_shape"])
            self.fix_step_size = True
        else:  # Random (bounded) value (independent for each component)
            self.fix_step_size = False
            self.step_size_ = jnp.zeros(self.config["sample_shape"])

        # Density function
        self.D_cl = self.config["density_cl"]

        # Initialization
        # Number of current acceptances
        self.acceptances = 0
        self.mean_acc = 0
        # Flag to avoid extra computation of derivative
        # (true just in first dynamic step for first iter)
        self.first_derivative = True

        # Store samples
        self.qall = []
        self.qpath = []
        if "store_path" in self.config.keys():
            self.store_path = self.config["store_path"]
        else:
            self.store_path = False

    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform random initialization of required state variables."""
        # Initialize state
        self.q_ = samples.copy()
        # Initialize step size randomly (if requested)
        if not self.fix_step_size:
            # Randomly initialize step size for updating Hamiltonian dynamics
            key, subkey = jax.random.split(key)
            self.step_size_ = self.update_stepsize(subkey)
        return key

    def update_stepsize(
        self, key: ArrayLike, min_step_size: float = 1e-4, delta_step_size: float = 5e-3
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
        step_size = (
            min_step_size
            + jax.random.uniform(key, shape=self.config["sample_shape"]) * delta_step_size
        )
        return step_size

    def step(self, key: ArrayLike, prev_samples: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        """Compute one step of sampler."""
        numleapfrog = self.config["numleapfrog"]
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # Initialize momentum variables randomly
        self.p_rn_ = jax.random.normal(subkey1, shape=self.q_.shape)
        # Store old state
        q_old = self.q_.copy()
        p_rn_old = self.p_rn_.copy()
        self.Eold = self.compute_energy(q_old, p_rn_old)
        # Advance state via leapfrog discretization
        self.compute_leapfrog_step(numleapfrog)
        # Negate momentum variables p
        self.p_rn_ = -self.p_rn_
        # Compute new energy
        self.Enew = self.compute_energy(self.q_, self.p_rn_)
        # Metropolis accept/reject
        accept = accept_Metropolis(subkey2, self.Eold, self.Enew)
        self.q_ = accept * self.q_ + (1 - accept) * q_old
        self.p_rn_ = accept * self.p_rn_ + (1 - accept) * p_rn_old
        self.acceptances = self.acceptances + accept.sum()
        self.mean_acc = accept.mean()

        if not self.fix_step_size:
            # Update step size for Hamiltonian dynamics
            self.step_size_ = self.update_stepsize(subkey3)

        self.qall.append(self.q_)

        return key, self.q_

    def compute_energy(self, q: ArrayLike, p_rn: ArrayLike) -> ArrayLike:
        """Compute energy of system.

        The energy of the system is the sum of potential and kinetic
        energy. The potential energy is the logarithm of the unormalized
        posterior. The kinetic energy is half the sum of the square of
        the momentum of the system.

        Args:
            q: System state.
            p_rn: System momentum.

        Returns:
            Energy of the current system configuration.
        """
        potentialE = -self.D_cl.log_target_proposal(q, self.tempering)  # potential energy
        kineticE = jnp.sum(p_rn**2, axis=1) / 2.0  # kinetic energy
        return potentialE + kineticE

    def leapfrog_scan_body(self, st: Tuple[ArrayLike], dum: ArrayLike):
        """Innermost update of leapfrog algorithm.

        This function formats the innermost update of the leapfrog
        algorithm to be computed via :meth:`jax.lax.scan`. This is done
        to reduce computation time via the jit functionality of jax.

        Args:
            st: Current state described via the tuple :math:`(q^{(t)}, p^{(t)})`
             with :math:`q` state and :math:`p` momentum variables.
            dum: Dummy variable. `lax.scan` assumes that the function
                to scan has two parameters: a carry of the state and
                an auxiliary input xs. In this case, we only need the
                previous state, so the auxiliary variable is a dummy
                input to satisfy the :meth:`jax.lax.scan` format.

        Returns:
            Updated state via tuple :math:`(q^{(t+1)}, p^{(t+1)})` and
            the current slice :math:`q^{(t+1)}`. The latter is to gather
            and return the complete trajectory (if requested).
        """
        # q-step
        q = st[0] + self.step_size_ * st[1]
        # update der_q_
        q_der = -self.D_cl.der_log_target_proposal(q, self.tempering)
        # Half p-steps merged
        p = st[1] - self.step_size_ * q_der
        return (q, p), q

    def compute_leapfrog_step(self, numsteps: int = 1):
        """Advance state using leapfrog numerical discretization."""
        if self.first_derivative:  # First time that derivative is computed
            self.q_der_ = -self.D_cl.der_log_target_proposal(
                self.q_, self.tempering
            )  # update der_q
            self.first_der = False

        qpath = []
        if self.store_path:
            qpath.append(self.q_)
        # Initial half p-step
        self.p_rn_ = self.p_rn_ - self.step_size_ * self.q_der_ / 2.0

        # Core repetitions of leapfrog steps implemented via lax.scan
        (self.q_, self.p_rn_), qpath_ = jax.lax.scan(
            self.leapfrog_scan_body, (self.q_, self.p_rn_), xs=None, length=numsteps - 1
        )

        if self.store_path:
            qpath.extend(qpath_)

        # last q-step
        self.q_ = self.q_ + self.step_size_ * self.p_rn_
        if self.store_path:
            qpath.append(self.q_)
            self.qpath.append(qpath)

        # Final half p-step
        self.q_der_ = -self.D_cl.der_log_target_proposal(self.q_, self.tempering)  # update der_q_
        self.p_rn_ = self.p_rn_ - self.step_size_ * self.q_der_ / 2.0

    def print_stats(self):
        """Print statistics computed during sample generation."""
        print(
            f"Iter: {self.itnum:>5d}, acceptances: {self.acceptances:>7.6e}, Eold: {self.Eold[0]:>7.6e}, Enew: {self.Enew[0]:>7.6e}"
        )


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
        super().__init__(config)
        self.numsteps = self.config["numsteps"]
        # Unnormalized log-weights
        self.logw = jnp.log(jnp.ones(self.Nsamples) / self.Nsamples)
        # log of normalization constant
        self.logZ = 0.0

        # HMC Object -> sampler state is HMC state
        self.hmc_ = HMC(config)

        # Density function
        self.D_cl = self.config["density_cl"]
        # Effective sample size threshold for resampling
        self.ESS_threshold = self.config["ESS_thres"]
        self.ess = 1.0

    def post_initialization(self, key: ArrayLike, samples: ArrayLike) -> ArrayLike:
        """Perform random initialization of required state variables."""
        # Initialize HMC state
        self.hmc_.q_ = samples.copy()
        return key

    def step(
        self,
        key: ArrayLike,
        prev_samples: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Compute one step of sampler."""
        dlogtw = (
            self.D_cl.log_target_proposal(self.hmc_.q_, self.tempering_fn(self.itnum + 1))
            - self.D_cl.log_target_proposal(self.hmc_.q_, self.tempering_fn(self.itnum))
        ).squeeze()
        self.logw = self.logw + dlogtw

        self.ess = self.compute_ess()
        if self.ess < self.config["ESS_thres"]:
            self.logZ = self.logZ + jax.scipy.special.logsumexp(self.logw, axis=-1)
            key, subkey = jax.random.split(key)
            self.resample(subkey)
            print(f"!Resampled at tStep = {self.itnum}; logZ = {self.logZ:>13.10f}")

        # Advance sample via HMC
        samples = prev_samples
        self.hmc_.tempering = self.tempering  # Sync SMC and HMC tempering
        for i in range(self.numsteps):
            key, samples = self.hmc_.step(key, samples)

        return key, samples

    def compute_ess(self):
        """Compute effective sample size (ESS)."""
        first_term = 2.0 * jax.scipy.special.logsumexp(self.logw, axis=-1)
        second_term = jax.scipy.special.logsumexp(2.0 * self.logw, axis=-1)
        return jnp.exp(first_term - second_term) / self.Nsamples

    def resample(self, key: ArrayLike):
        """Resample particles."""
        index = jax.random.choice(
            key,
            jnp.arange(self.Nsamples),
            shape=(self.Nsamples,),
            replace=True,
            p=jax.nn.softmax(self.logw, axis=-1),
        )
        self.hmc_.q_ = self.hmc_.q_[index, :]
        self.logw = jnp.log(jnp.ones(self.Nsamples) / self.Nsamples)

    def print_stats(self):
        """Print statistics computed during sample generation."""
        logZ = self.logZ + jax.scipy.special.logsumexp(self.logw, axis=-1)
        print(
            f"Iter: {self.itnum:>5d}, fraction_acceptances: {self.hmc_.mean_acc:>11.10f}, ESS: {self.ess:>13.10f}, logZ: {logZ:>13.10f}"
        )

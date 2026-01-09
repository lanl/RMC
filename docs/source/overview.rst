Implemented Samplers
====================

Markov Chain Monte Carlo (MCMC)
-------------------------------

MCMC is a simulation-based method that constructs an ergodic Markov kernel
with invariant distribution using Metropolis–Hastings (MH) steps and Gibbs
moves :cite:`robert-2004-mcmc`.

Hamiltonian Monte Carlo (HMC)
-----------------------------

HMC is a method that introduces fictitious "momentum" variables that can be used to
produce distant proposals for the Metropolis algorithm, thereby avoiding the slow
exploration of the state space that results from the diffusive behaviour of simple
random-walk proposals :cite:`neal-2012-hmc`.

Sequential Monte Carlo (SMC)
----------------------------

SMC is a simulation-based method that samples from a sequence of probability
distributions which are approximated by a cloud of weighted random samples
which are propagated over time by using sequential Monte Carlo :cite:`delmoral-2006-smc`.


Stein Variational Gradient Descent (SVGD)
-----------------------------------------

The SVGD method transports a set of particles to match
the target distribution by applying a form of functional
gradient descent that minimizes the KL divergence :cite:`liu-2016-svgd`.


Liouville Flow with Importance Sampling (LFIS)
----------------------------------------------

The LFIS method learns a time-dependent velocity field that
deterministically transports samples from a simple initial
distribution to a complex target distribution, guided by a
prescribed path of annealed distributions. LFIS uses a
neural network to model the velocity field. This is trained
by enforcing the structure of a derived partial differential
equation :cite:`tian-2024-lfis`.

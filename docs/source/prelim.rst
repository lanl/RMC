Preliminaries
=============

Dependencies
------------

Python packages
    - Jax
    - Flax

Installation
------------

To use RMC (...)


JAX
===

This package uses `JAX <https://docs.jax.dev/en/latest/>`_ to integrate an array of classical and
generative modeling-driven samplers. We selected JAX for its advanced features,
such as composable transformations, just-in-time compilation, seamless CPU/GPU deployment
and distributed data parallelism. The package leverages JAX’s automatic differentiation to
enhance the efficiency of gradient-based samplers.


Machine Learning
================

Machine learning models are implemented in `Flax <https://flax.readthedocs.io/en/stable/>`_ and
include model definition and training functionality.

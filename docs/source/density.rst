Unnormalized Density
====================

For ease of description we follow the notation in
:cite:`tian-2024-lfis` and group the unnormalized
density to sample from in two different types described
next.


Type 1 - Density Path
---------------------

The time-dependent target density function,
:math:`\tilde{\rho}_{*}(x, t)`, for type 1 is usually
represented as

.. math:: \tilde{\rho}_{*}(x, t) = \mu^{1 - \tau(t)}(x) \; \tilde{\nu}^{\tau(t)}(x) \,,

with :math:`\tilde{\nu}` an unnormalized hard-to-sample
density function that is the target density to sample from,
:math:`\mu` an initial easy-to-sample distribution, and a
*schedule function* :math:`\tau`, representing a monotonic
function transforming time :math:`t` and satisfying
:math:`\tau(0) = 0` and :math:`\tau(1) = 1`.

The path in the type 1 time-dependent target density
function represents the gradual deformation of :math:`\mu`
into :math:`\tilde{\nu}`.


Type 2 - Bayesian Posterior
---------------------------

The time-dependent target density function for type 2
is usually represented as

.. math:: \tilde{\rho}_{*}(x, t) = L^{\tau(t)}(x) \; \pi(x)  \,,

with :math:`L` the likelihood function, :math:`\pi` the
prior density function, and :math:`\tau` the schedule
function (see definition in type 1).

The posterior in the type 2 time-dependent target density
function corresponds to the direct application of Bayes
theorem with a tempered likelihood.

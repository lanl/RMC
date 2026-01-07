Distributions
=============

For ease of notation we group the distributions to sample
from in two types described next.


Type 1 - Unnormalized Density Function Sampling
-----------------------------------------------

The time-dependent target density function,
:math:`\tilde{\rho}_{*}(x, t)`, for type 1 is usually
represented as

.. math:: \tilde{\rho}_{*}(x, t) = \mu^{1 - \tau(t)}(x) \; \tilde{\nu}^{\tau(t)}(x) \,,

with :math:`\tilde{\nu}` the unnormalized density function
to sample from, :math:`\mu` a base tractable
(easy-to-sample) distribution, and a *schedule function*
:math:`\tau`, representing a monotonic function transforming
time :math:`t` and satisfying :math:`\tau(0) = 0`
and :math:`\tau(1) = 1`.


Type 2 - Bayesian Posterior Sampling
------------------------------------

For type 2, the time-dependent target density function
is usually represented as

.. math:: \tilde{\rho}_{*}(x, t) = L^{\tau(t)}(x) \; \pi(x)  \,,

with :math:`L` the likelihood function, :math:`\pi` the
prior density function, and :math:`\tau` the schedule
function (see definition in type 1).

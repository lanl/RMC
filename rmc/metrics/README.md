# Sampling Metrics

This package implements numerical metrics for comparing collections of samples, developed for the RMC sampling meta-analysis. The metrics operate independently of sampler implementations and target-density interfaces.

## Conventions

Let

$$
X=\{x_i\}_{i=1}^N,\qquad
Y=\{y_j\}_{j=1}^M,
\qquad x_i,y_j\in\mathbb R^d,
$$

denote generated and reference samples, respectively. Their empirical probability measures are

$$
\widehat\mu_X=\frac1N\sum_{i=1}^N\delta_{x_i},
\qquad
\widehat\mu_Y=\frac1M\sum_{j=1}^M\delta_{y_j}.
$$

All sample arrays have shape `(N, d)` or `(M, d)`. Samples carry uniform empirical weights. Wasserstein, sliced Wasserstein, and Sinkhorn currently require equal sample counts, while MMD allows unequal counts. The implementations primarily use JAX and return scalar JAX arrays. Exceptions to JAX compilation support are noted below.

## API summary

| Function | Module | Default parameters | Output |
|---|---|---|---|
| `mmd_squared` | `mmd.py` | Gaussian kernel, length scale 1 | Unbiased estimate of MMD squared |
| `sliced_wasserstein` | `sliced_wasserstein.py` | p=2, 128 projections | Monte Carlo sliced Wasserstein distance |
| `wasserstein` | `wasserstein.py` | p=2 | Exact empirical Wasserstein distance |
| `sinkhorn_divergence` | `sinkhorn.py` | epsilon=0.1 | Debiased Sinkhorn divergence |
| `observable_error` | `observables.py` | relative=False | Error in an empirical expectation |
| `moment_errors` | `observables.py` | relative=False | First- and second-moment errors |
| `fisher_score_discrepancy` | `scores.py` | Target and approximate scores | Fisher divergence |
| `mean_log_density` | `density.py` | Target log density | Empirical mean log density |
| `log_density_difference` | `density.py` | Target log density | Signed difference of mean log densities |
| `log_density_error` | `density.py` | Target log density | Absolute difference of mean log densities |
| `mode_region_probabilities` | `modes.py` | centers, radius | K+1 empirical region probabilities |
| `discrete_tv` | `modes.py` | centers, radius | TV across K+1 regions |
| `mode_coverage` | `modes.py` | centers, radius | Fraction of prescribed modes occupied |
| `intermode_mass` | `modes.py` | centers, radius | Complementary-region mass |
| `importance_ess` | `importance.py` | log_weights=False | Weight-based ESS |
| `mcmc_ess` | `mcmc.py` | Scalar time series | Observable autocorrelation ESS |

All functions can be imported from their respective modules under `rmc.metrics`.

---

## 1. Maximum Mean Discrepancy (MMD)

**Module:** `rmc.metrics.mmd`

```python
mmd_squared(
    samples,
    reference_samples,
    kernel="gaussian",
    length_scale=1.0,
    nu=1.5,
)
```

### Mathematical definition

For a positive-definite kernel $k$, the squared population MMD is

$$
\operatorname{MMD}_k^2(\mu,\nu)
=
\mathbb E_{X,X'\sim\mu}[k(X,X')]
+
\mathbb E_{Y,Y'\sim\nu}[k(Y,Y')]
-
2\mathbb E_{X\sim\mu,Y\sim\nu}[k(X,Y)].
$$

The implementation uses the unbiased U-statistic estimator:

$$
\boxed{
\begin{aligned}
\widehat{\operatorname{MMD}}_u^2(X,Y)
={}&
\frac{1}{N(N-1)}
\sum_{i\ne j}k(x_i,x_j)
\\
&+
\frac{1}{M(M-1)}
\sum_{i\ne j}k(y_i,y_j)
\\
&-
\frac{2}{NM}
\sum_{i=1}^N\sum_{j=1}^M k(x_i,y_j).
\end{aligned}
}
$$

### Supported kernels

**Gaussian RBF:**

$$
k_\ell(x,y)
=
\exp\left(-\frac{\|x-y\|^2}{2\ell^2}\right).
$$

**Matérn:** Let $r=\|x-y\|/\ell$. The supported smoothness parameters are

$$
\begin{aligned}
k_{1/2}(r)&=e^{-r},\\
k_{3/2}(r)&=(1+\sqrt3\,r)e^{-\sqrt3\,r},\\
k_{5/2}(r)&=
\left(1+\sqrt5\,r+\frac53r^2\right)e^{-\sqrt5\,r}.
\end{aligned}
$$

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `samples` | ArrayLike | Generated samples, shape `(N, d)` |
| `reference_samples` | ArrayLike | Reference samples, shape `(M, d)` |
| `kernel` | str | `"gaussian"` or `"matern"` |
| `length_scale` | float | Positive kernel length scale; default 1 |
| `nu` | float | Matérn smoothness: 0.5, 1.5, or 2.5; default 1.5 |

Both collections must contain at least two samples.

### Numerical properties

- The estimator is unbiased for population MMD squared under
  independent sampling.

- Negative estimates are possible and are preserved without clipping.
- The return value is squared MMD, not its square root.
- The estimator uses full pairwise kernel matrices, with
  $O(N^2+M^2+NM)$ memory requirements.

- Kernel bandwidth affects the numerical result and must be recorded
  in benchmark configurations.

### Example

```python
from rmc.metrics.mmd import mmd_squared
value = mmd_squared(
    samples,
    reference_samples,
    kernel="matern",
    length_scale=1.0,
    nu=1.5,
)
```

---

## 2. Sliced Wasserstein Distance

**Module:** `rmc.metrics.sliced_wasserstein`

```python
sliced_wasserstein(
    samples,
    reference_samples,
    key,
    p=2.0,
    num_projections=128,
)
```

### Mathematical definition

For a unit vector $\theta\in S^{d-1}$, let $P_\theta(x)=\langle\theta,x\rangle$. The population sliced Wasserstein distance is

$$
\operatorname{SW}_p(\mu,\nu)
=
\left[
\int_{S^{d-1}}
W_p^p\left(
(P_\theta)_\#\mu,
(P_\theta)_\#\nu
\right)
\,d\sigma(\theta)
\right]^{1/p},
$$

where $\sigma$ is the uniform probability measure on the sphere. For equal sample counts, the empirical one-dimensional Wasserstein cost is computed exactly by sorting the projected observations. With independent random directions $\theta_1,\ldots,\theta_L\sim\sigma$, the implemented estimator is

$$
\boxed{
\widehat{\operatorname{SW}}_p(X,Y)
=
\left[
\frac{1}{LN}
\sum_{\ell=1}^L\sum_{i=1}^N
\left\|
(\theta_\ell^\top X)_{(i)}
-
(\theta_\ell^\top Y)_{(i)}
\right\|^p
\right]^{1/p}.
}
$$

Here the subscript $(i)$ denotes the $i$-th sorted projected value.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `samples` | ArrayLike | Generated samples, shape `(N, d)` |
| `reference_samples` | ArrayLike | Reference samples, shape `(N, d)` |
| `key` | JAX PRNG key | Random projection generation |
| `p` | float | Wasserstein order, at least 1; default 2 |
| `num_projections` | int | Number of random directions; default 128 |

### Numerical properties

- Both sample collections must have matching shapes.
- Projection directions are obtained by normalizing Gaussian vectors.
- Identical random keys reproduce the same projections.
- Sorting computes the exact one-dimensional empirical transport cost
  for each projection.

- The spherical integral is approximated by Monte Carlo.
- The returned distance includes the $p$-th root.
- The implementation is compatible with `jax.jit` for fixed shapes
  and static parameters such as the number of projections. The quantity inside the $p$-th root is an unbiased Monte Carlo estimate of the spherical average of the projected empirical costs. The distance itself is generally not an unbiased estimator of the population sliced Wasserstein distance.

### Example

```python
import jax
from rmc.metrics.sliced_wasserstein import sliced_wasserstein
key = jax.random.key(42)
value = sliced_wasserstein(
    samples,
    reference_samples,
    key,
    p=2.0,
    num_projections=128,
)
```

---

## 3. Exact Empirical Wasserstein Distance

**Module:** `rmc.metrics.wasserstein`

```python
wasserstein(
    samples,
    reference_samples,
    p=2.0,
)
```

### Mathematical definition

For equally weighted empirical measures of equal size,

$$
\boxed{
W_p(\widehat\mu_X,\widehat\mu_Y)
=
\left[
\min_{\pi\in S_N}
\frac1N\sum_{i=1}^N
\|x_i-y_{\pi(i)}\|^p
\right]^{1/p},
}
$$

where $S_N$ denotes the permutations of $\{1,\ldots,N\}$.

### Numerical method

For $d=1$, the exact solution is obtained by sorting:

$$
W_p^p(\widehat\mu_X,\widehat\mu_Y)
=
\frac1N\sum_{i=1}^N
|x_{(i)}-y_{(i)}|^p.
$$

For $d>1$, construct the cost matrix

$$
C_{ij}=|x_i-y_j|^p
$$

and solve the linear assignment problem using `scipy.optimize.linear_sum_assignment`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `samples` | ArrayLike | Generated samples, shape `(N, d)` |
| `reference_samples` | ArrayLike | Reference samples, shape `(N, d)` |
| `p` | float | Wasserstein order, finite and at least 1; default 2 |

### Numerical properties

- This computes exact discrete optimal transport up to numerical
  precision, rather than an entropically regularized approximation.

- The returned value is $W_p$, including the $p$-th root.
- Multidimensional computation uses $O(N^2)$ cost-matrix storage.
- The assignment solver runs on the CPU and is not JAX-JIT-compatible.
- SciPy is currently available through JAX's dependency chain but is
  not explicitly declared in RMC's `pyproject.toml`. The exact multidimensional assignment problem may become expensive for large sample collections.

### Example

```python
from rmc.metrics.wasserstein import wasserstein
value = wasserstein(
    samples,
    reference_samples,
    p=2.0,
)
```

---

## 4. Sinkhorn Divergence

**Module:** `rmc.metrics.sinkhorn`

```python
sinkhorn_divergence(
    samples,
    reference_samples,
    epsilon=0.1,
    tolerance=1e-5,
    max_iterations=1000,
)
```

### Mathematical definition

Let $a=b=\frac1N\mathbf1$, and define the squared Euclidean transport cost

$$
C_{ij}=|x_i-y_j|^2.
$$

The admissible couplings satisfy

$$
U(a,b)
=
\{P\geq0:P\mathbf1=a,\ P^\top\mathbf1=b\}.
$$

Define the entropically regularized transport objective by

$$
\operatorname{OT}_\varepsilon(X,Y)
=
\min_{P\in U(a,b)}
\left[
\langle P,C\rangle
+
\varepsilon\operatorname{KL}(P\|ab^\top)
\right],
$$

where

$$
\operatorname{KL}(P\|ab^\top)
=
\sum_{i,j}P_{ij}
\log\frac{P_{ij}}{a_i b_j}.
$$

The implemented metric is the debiased Sinkhorn divergence:

$$
\boxed{
S_\varepsilon(X,Y)
=
\operatorname{OT}_\varepsilon(X,Y)
-\frac12\operatorname{OT}_\varepsilon(X,X)
-\frac12\operatorname{OT}_\varepsilon(Y,Y).
}
$$

### Numerical method

The implementation performs three regularized transport solves, using log-domain Sinkhorn iterations. The Gibbs kernel is

$$
K_{ij}=\exp(-C_{ij}/\varepsilon),
$$

and the coupling has the scaling form

$$
P=\operatorname{diag}(u)K\operatorname{diag}(v).
$$

Updates are carried out in logarithmic coordinates using JAX's `logsumexp` to improve numerical stability. The convergence residual is

$$
r=
\max\left\{
\|P\mathbf1-a\|_\infty,
\|P^\top\mathbf1-b\|_\infty
\right\}.
$$

The solver terminates when $r\leq\texttt{tolerance}$ or the iteration limit is reached. Failure to reach tolerance raises `RuntimeError`. The returned objective includes both the transport cost and the KL regularization term.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `samples` | ArrayLike | Generated samples, shape `(N, d)` |
| `reference_samples` | ArrayLike | Reference samples, shape `(N, d)` |
| `epsilon` | float | Positive regularization strength; default 0.1 |
| `tolerance` | float | Absolute marginal residual tolerance; default 1e-5 |
| `max_iterations` | int | Maximum iterations per solve; default 1000 |

### Numerical properties

- The ground cost is squared Euclidean distance.
- The output has squared-distance units.
- No square root is applied.
- The divergence vanishes for identical empirical measures, up to
  numerical error.

- The divergence approaches squared empirical Wasserstein distance
  as the regularization parameter tends to zero, when the solves   are sufficiently accurate.

- Smaller regularization can lead to slow convergence.
- Nonconvergence raises an exception rather than silently returning
  the last iterate.

- Memory usage is $O(N^2)$.
- Because convergence checks convert JAX scalars to Python values,
  the public function is not directly JAX-JIT-compatible. The parameter `epsilon` is expressed in the same units as the squared-distance ground cost. Its value must be recorded in benchmark configurations. A small marginal residual alone does not provide a universal bound on the error in the regularized objective.

### Example

```python
from rmc.metrics.sinkhorn import sinkhorn_divergence
value = sinkhorn_divergence(
    samples,
    reference_samples,
    epsilon=0.5,
    tolerance=1e-5,
    max_iterations=1000,
)
```

---

## 5. Errors of Observables and Moments

**Module:** `rmc.metrics.observables`

```python
observable_error(
    samples,
    reference_samples,
    observable,
    relative=False,
)
moment_errors(
    samples,
    reference_samples,
    relative=False,
)
```

### Mathematical definitions

For an observable $f:\mathbb R^d\to\mathbb R^k$, define

$$
\widehat{\mathbb E}_X[f]
=
\frac1N\sum_{i=1}^N f(x_i),
\qquad
\widehat{\mathbb E}_Y[f]
=
\frac1M\sum_{j=1}^M f(y_j).
$$

The absolute error is

$$
E_f^{\mathrm{abs}}
=
\left\|
\widehat{\mathbb E}_X[f]
-
\widehat{\mathbb E}_Y[f]
\right\|.
$$

The relative error is

$$
E_f^{\mathrm{rel}}
=
\frac{
\left\|\widehat{\mathbb E}_X[f]
-\widehat{\mathbb E}_Y[f]\right\|
}{
\left\|\widehat{\mathbb E}_Y[f]\right\|
}.
$$

Absolute error uses absolute value for scalar observables, Euclidean norm for vectors, and Frobenius norm for matrices. When the reference norm is zero, relative error is defined as zero if the discrepancy is zero and infinity otherwise. The reference is an empirical expectation, not necessarily an analytically known target expectation. The resulting discrepancy therefore includes reference sampling error.

### First and second moments

The dedicated `moment_errors` function evaluates

$$
\widehat m_X=\frac1N\sum_{i=1}^N x_i,
\qquad
\widehat M_X=\frac1N\sum_{i=1}^N x_ix_i^\top,
$$

with analogous definitions for the reference samples. It returns

$$
\begin{aligned}
E_{\mathrm{mean}}
&=\|\widehat m_X-\widehat m_Y\|_2,\\
E_{\mathrm{second}}
&=\|\widehat M_X-\widehat M_Y\|_F.
\end{aligned}
$$

These are raw second-moment errors, not covariance errors. When `relative=True`, each error is divided by the norm of its corresponding reference moment, using the same zero-denominator convention as `observable_error`.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `samples` | ArrayLike | Generated samples, shape `(N, d)` |
| `reference_samples` | ArrayLike | Reference samples, shape `(M, d)` |
| `observable` | callable | Function evaluated on one sample of shape `(d,)` |
| `relative` | bool | Return relative rather than absolute error; default False |

Unequal sample counts are supported. The sample dimensions must match and both collections must be nonempty.

### Returns

`observable_error` returns a scalar JAX array. `moment_errors` returns a dictionary of scalar JAX arrays:

```python
{
    "mean": ...,
    "second_moment": ...,
}
```

### Numerical properties

- Observables are evaluated using `jax.vmap`.
- The supplied observable must be compatible with JAX transformations.
- Both functions can be used with `jax.jit` for suitable observables
  and fixed input shapes.

- All errors are nonnegative, apart from possible numerical anomalies.
- Relative errors can be infinite when the reference expectation
  vanishes.

- No target log density or score is required.
- When a log-density evaluator is available, its values can be
  compared through `observable_error`.

- Nonfinite observable evaluations propagate into the returned error.
- The choice of observable and reference distribution must be
  recorded in benchmark configurations.

### Examples

```python
import jax.numpy as jnp
from rmc.metrics.observables import moment_errors, observable_error
# Compare first and raw second moments.
errors = moment_errors(samples, reference_samples)
# Compare the expectation of squared radius.
radius_error = observable_error(
    samples,
    reference_samples,
    lambda x: jnp.sum(x**2),
)
# Relative error in a scalar observable.
relative_error = observable_error(
    samples,
    reference_samples,
    lambda x: x[0],
    relative=True,
)
```

For a vector-valued observable, the returned scalar is the norm of the difference between estimated expectation vectors. Individual coordinate errors can instead be obtained by supplying coordinate-specific scalar observables.

### Validation

Tests are located in `rmc/test/test_observables.py`. They include analytical moment examples, relative errors, zero reference values, unequal sample counts, permutation invariance, and JAX compilation.

---

## 6. Discrete Total Variation, Mode Coverage, and Inter-Mode Mass

**Module:** `rmc.metrics.modes`

```python
mode_region_probabilities(samples, centers, radius)
discrete_tv(samples, reference_samples, centers, radius)
mode_coverage(samples, centers, radius)
intermode_mass(samples, centers, radius)
```

### Mathematical definitions

Let $m_0,\ldots,m_{K-1}\in\mathbb R^d$ denote prescribed mode centers. For a fixed positive radius $r$, assume the closed balls

$$
B_k=\{x\in\mathbb R^d:\|x-m_k\|\leq r\}, \qquad k=0,\ldots,K-1,
$$

are pairwise disjoint. The complementary region is

$$
B_K=\mathbb R^d\setminus\bigcup_{k=0}^{K-1}B_k.
$$

For the nine-Gaussian benchmark, this defines ten regions: nine mode balls and the remaining inter-mode region. Define empirical region probabilities by

$$
\widehat p_k=\frac1N\sum_{i=1}^N\mathbf1_{B_k}(x_i),\qquad
\widehat q_k=\frac1M\sum_{j=1}^M\mathbf1_{B_k}(y_j),\quad k=0,\ldots,K.
$$

The region-based total variation distance is

$$
\widehat{\mathrm{TV}}(X,Y)=\frac12\sum_{k=0}^K|\widehat p_k-\widehat q_k|.
$$

The coverage of prescribed modes and inter-mode mass are

$$
\operatorname{Coverage}(X)=\frac1K\sum_{k=0}^{K-1}\mathbf1\{\widehat p_k>0\},
\qquad \operatorname{IntermodeMass}(X)=\widehat p_K.
$$

### Parameters and returns

| Function | Input | Output |
|---|---|---|
| `mode_region_probabilities` | Samples `(N,d)`, centers `(K,d)`, radius | JAX array `(K+1,)`; last entry is complementary mass |
| `discrete_tv` | Generated/reference samples, centers, radius | Scalar TV across all `K+1` regions |
| `mode_coverage` | Samples, centers, radius | Fraction of the `K` mode balls containing at least one sample |
| `intermode_mass` | Samples, centers, radius | Fraction of samples outside all mode balls |

The radius and centers must be fixed across sampler comparisons. A point on a ball boundary belongs to that ball. The API currently requires a shared radius and disjoint balls; invalid geometry raises `ValueError`. All public functions perform host-side geometry validation and are not directly JIT-compatible. The diagnostics depend on the sample size and partition choice.

### Example

```python
import jax.numpy as jnp
from rmc.metrics.modes import (
    discrete_tv, intermode_mass, mode_coverage, mode_region_probabilities,
)

centers = jnp.array([
    [-4., -4.], [0., -4.], [4., -4.],
    [-4., 0.], [0., 0.], [4., 0.],
    [-4., 4.], [0., 4.], [4., 4.],
])
radius = 1.0  # Illustrative: benchmark protocols must specify their own value.
probabilities = mode_region_probabilities(samples, centers, radius)
tv = discrete_tv(samples, reference_samples, centers, radius)
coverage = mode_coverage(samples, centers, radius)
outside_mass = intermode_mass(samples, centers, radius)
```

### Validation

The tests in `rmc/test/test_modes.py` cover the ten-region partition, exact TV values, coverage, complementary mass, boundary behavior, and invalid geometry.

---

## 7. Fisher Score Discrepancy

**Module:** `rmc.metrics.scores`

```python
fisher_score_discrepancy(
    samples,
    target_score,
    approximate_score,
)
```

### Mathematical definition

Let $\pi$ be the target density and $q$ an approximate density. Their score functions are

$$
s_\pi(x)=\nabla\log\pi(x),
\qquad
s_q(x)=\nabla\log q(x).
$$

We adopt the Fisher divergence convention

$$
D_F(q\|\pi)
=
\frac12\mathbb E_{X\sim q}
\left[\|s_q(X)-s_\pi(X)\|_2^2\right].
$$

Given samples $x_1,\ldots,x_N$ from $q$, the estimator is

$$
\boxed{
\widehat D_F(q\|\pi)
=
\frac{1}{2N}\sum_{i=1}^N
\|s_q(x_i)-s_\pi(x_i)\|_2^2.
}
$$

The direction of the expectation matters: this definition integrates with respect to the approximate distribution $q$.

### Parameters

| Parameter | Description |
|---|---|
| `samples` | Samples from q, shape `(N, d)` |
| `target_score` | Callable mapping `(d,)` to `(d,)` |
| `approximate_score` | Callable mapping `(d,)` to `(d,)` |

### Properties

- The output is a nonnegative scalar JAX array.
- Both scores are evaluated pointwise using `jax.vmap`.
- With JAX-compatible score functions, the estimator supports `jax.jit`.
- Both score functions must be available; samples alone are insufficient.
- Unnormalized density functions may be differentiated to obtain scores.
- Nonfinite score evaluations propagate into the result.
- The factor 1/2 is part of our chosen convention.
For $q=\mathcal N(\mu,I)$ and $\pi=\mathcal N(0,I)$,

$$
D_F(q\|\pi)=\frac12\|\mu\|_2^2.
$$

### Example

```python
import jax
import jax.numpy as jnp
from rmc.metrics.scores import fisher_score_discrepancy
def target_log_density(x):
    return -0.5 * jnp.sum(x**2)
def approximate_log_density(x):
    return -0.5 * jnp.sum((x - 1.0)**2)
value = fisher_score_discrepancy(
    samples,
    target_score=jax.grad(target_log_density),
    approximate_score=jax.grad(approximate_log_density),
)
```

### Validation

Tests are located in `rmc/test/test_scores.py` and cover Gaussian examples, exact score matching, JAX autodifferentiation, JIT compatibility, permutation invariance, and invalid shapes.

---

## 8. Log-Density Diagnostics

**Module:** `rmc.metrics.density`

### API

```python
mean_log_density(samples, log_density)
log_density_difference(
    samples,
    reference_samples,
    log_density,
)
log_density_error(
    samples,
    reference_samples,
    log_density,
)
```

### Mathematical definitions

For a scalar target log density $\ell(x)=\log\widetilde\pi(x)$, define

$$
L_X=\frac1N\sum_{i=1}^N\ell(x_i),
\qquad
L_Y=\frac1M\sum_{j=1}^M\ell(y_j).
$$

The signed difference is

$$
\boxed{\Delta_\ell(X,Y)=L_X-L_Y}
$$

and the absolute error is

$$
\boxed{E_\ell(X,Y)=|\Delta_\ell(X,Y)|.}
$$

Since any additive constant cancels in the difference, the comparison is invariant to the unknown log normalizing constant of an unnormalized target density. The absolute mean log density itself does depend on the normalization convention.

### Parameters

| Parameter | Description |
|---|---|
| `samples` | Generated samples, shape `(N, d)` |
| `reference_samples` | Reference samples, shape `(M, d)` |
| `log_density` | Callable mapping `(d,)` to a scalar |

Unequal sample counts are supported.

### Properties

- All three functions return scalar JAX arrays.
- Log densities are evaluated pointwise using `jax.vmap`.
- JAX-compatible log-density functions support `jax.jit`.
- Signed and absolute comparisons are invariant to additive
  constants in the supplied log density.

- Nonfinite log-density evaluations propagate into the result.
- The discrepancy can be computed from an unnormalized target.
- The reference mean has its own Monte Carlo sampling error.
- Mean log density alone is not a distributional discrepancy:
  concentrating on high-density regions may improve the mean   log density while reducing mode coverage.

- Relative errors are intentionally omitted because the relative
  normalization would depend on an arbitrary additive constant.

### Example

```python
import jax.numpy as jnp
from rmc.metrics.density import (
    mean_log_density,
    log_density_difference,
    log_density_error,
)
def log_target(x):
    return -0.5 * jnp.sum(x**2)
generated_mean = mean_log_density(samples, log_target)
signed_difference = log_density_difference(
    samples,
    reference_samples,
    log_target,
)
absolute_error = log_density_error(
    samples,
    reference_samples,
    log_target,
)
```

### Validation

Tests are located in `rmc/test/test_density.py` and cover analytical examples, signed differences, additive-constant invariance, unequal sample counts, JIT compatibility, and agreement with the generic observable-error implementation.

---

## Validation

Run all metric tests from the repository root:

```bash
python -m pytest \
  rmc/test/test_mmd.py \
  rmc/test/test_sliced_wasserstein.py \
  rmc/test/test_wasserstein.py \
  rmc/test/test_sinkhorn.py \
  rmc/test/test_observables.py \
  rmc/test/test_modes.py \
  rmc/test/test_scores.py \
  rmc/test/test_density.py -q
```

Validation includes analytical examples, numerical comparisons, permutation invariance, input validation, and relevant JAX compilation checks.

## Benchmark reporting conventions

Record the sample count, dimension, reference-sample construction, metric parameter values, random keys and projection counts, Sinkhorn tolerances and iteration budget, definitions of mode balls, and evaluation failures. Different metric units and parameter choices should be reported separately.

## References

- Gretton et al. (2012), *A Kernel Two-Sample Test*. https://www.jmlr.org/papers/v13/gretton12a.html
- Rabin et al. (2011), *Wasserstein Barycenter and Its Application to Texture Mixing*.
- Cuturi (2013), *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. https://arxiv.org/abs/1306.0895
- Feydy et al. (2019), *Interpolating between Optimal Transport and MMD using Sinkhorn Divergences*. https://arxiv.org/abs/1810.08278

---

## 9. Importance-Sampling Effective Sample Size

**Module:** `rmc.metrics.importance`

### API

```python
importance_ess(weights, log_weights=False)
```

### Mathematical definition

For nonnegative weights $w_1,\ldots,w_N$, define

$$
\boxed{
\operatorname{ESS}_{\mathrm{IS}}
=
\frac{\left(\sum_{i=1}^N w_i\right)^2}
{\sum_{i=1}^N w_i^2}.
}
$$

Equivalently, using normalized weights

$$
\bar w_i=\frac{w_i}{\sum_j w_j},
\qquad
\operatorname{ESS}_{\mathrm{IS}}
=\frac{1}{\sum_i\bar w_i^2}.
$$

For log weights $\ell_i=\log w_i$, the implementation evaluates

$$
\operatorname{ESS}_{\mathrm{IS}}
=
\exp\left[
2\operatorname{LSE}(\ell)
-\operatorname{LSE}(2\ell)
\right],
$$

where $\operatorname{LSE}$ denotes log-sum-exp.

### Parameters

| Parameter | Description |
|---|---|
| `weights` | One-dimensional array of weights or log weights |
| `log_weights` | If `True`, interpret the input as log weights |

### Properties

- Returns a scalar JAX array in $[1,N]$.
- Uniform weights give ESS equal to $N$.
- One nonzero weight gives ESS equal to 1.
- ESS is invariant to positive rescaling of weights.
- Log weights may contain `-inf` for zero-weight particles.
- Invalid, negative, or all-zero ordinary weights are rejected.
- This is a weight-degeneracy diagnostic, not MCMC ESS.
- Input validation prevents direct JIT compilation of the public function.

### Example

```python
import jax.numpy as jnp
from rmc.metrics.importance import importance_ess

ess = importance_ess(jnp.array([1.0, 2.0, 3.0]))

log_ess = importance_ess(
    jnp.array([-1000.0, -999.0, -998.0]),
    log_weights=True,
)
```

Tests: `rmc/test/test_importance.py`.

---

## 10. MCMC Effective Sample Size

**Module:** `rmc.metrics.mcmc`

### API

```python
mcmc_ess(chain)
```

### Mathematical definition

For an observable $f$ of a stationary Markov chain, let $Y_t=f(X_t)$. The scalar time series has lag-$k$
autocorrelation $\rho_k$, the integrated autocorrelation time is

$$
\tau_{\mathrm{int}}
=
1+2\sum_{k=1}^{\infty}\rho_k.
$$

The effective sample size is

$$
\boxed{
\operatorname{ESS}_{\mathrm{MCMC}}
=
\frac{N}{\tau_{\mathrm{int}}}.
}
$$

The implementation estimates autocorrelations using the FFT
and biased sample autocovariances.

It forms paired autocorrelation sums

$$
\widehat\Gamma_j
=
\widehat\rho_{2j}+\widehat\rho_{2j+1},
$$

retains their initial positive sequence, and replaces the
retained values by their running minima to impose monotonicity.

The resulting truncated estimate is

$$
\widehat\tau_{\mathrm{int}}
=
2\sum_{j=0}^{J}\widetilde\Gamma_j-1.
$$

For reporting, the implementation clips the integrated
autocorrelation time to $[1,N]$, so the returned ESS lies in
$[1,N]$ for nonconstant coordinates.

### Parameters

| Parameter | Description |
|---|---|
| `observations` | Ordered scalar observable values, shape `(N,)` |

At least four draws are required.

### Returns

A scalar JAX array containing the estimated effective sample size.
A constant observable time series returns `NaN`.

### Properties and limitations

- The order of samples matters.
- The estimator assumes stationarity and sufficiently long chains.
- Each call evaluates one scalar observable.
- Vector-valued observations must first be reduced to a scalar observable.
- It is not a convergence diagnostic or a substitute for split-R-hat.
- Independent chains must not be concatenated.
- Negative correlation can theoretically produce ESS greater than
  $N$; this implementation intentionally caps the reported value at $N$.
- Computation uses NumPy FFTs on the CPU and is not JAX-JIT-compatible.
- MCMC ESS and importance-weight ESS are distinct statistical quantities.

### Example

```python
from rmc.metrics.mcmc import mcmc_ess

import jax.numpy as jnp

# samples: ordered Markov chain of shape (N, d)
energy = jnp.sum(samples**2, axis=1)
energy_ess = mcmc_ess(energy)
```

Tests: `rmc/test/test_mcmc.py`.

### Reference

Geyer (1992), *Practical Markov Chain Monte Carlo*,
Statistical Science, 7(4), 473–483.

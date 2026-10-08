# ASBS Reference Implementation

## Paper

Adjoint Schrödinger Bridge Sampler

arXiv:2506.22565

## Official repository

https://github.com/facebookresearch/adjoint_samplers

## Reference repository state

Commit: `a972860`
Branch: `main`

## RMC base

Repository: https://github.com/lanl/RMC
Branch: `develop`
Commit at porting start: `7ac94c3`

## RMC development branch

Local branch: `asbs-dev`
Based on: `origin/develop`

## Implementation selection

Every ASBS configuration must explicitly set `asbs_implementation` to
one of the following values:

- `paper`: direct numerical realization of the main-paper ASBS algorithm,
  without practical solver features that are not part of the core
  algorithm.
- `official_repository`: practical implementation corresponding as
  closely as possible to the procedure and settings used in the official
  ASBS repository and associated experiments.

The selector identifies a coherent reproducibility profile rather than a
collection of independently enabled repository features. No implementation
is selected implicitly.

## Purpose

This file records the exact external reference state used for the RMC ASBS port. The port should preserve a distinction between the mathematical algorithm described in the paper and additional practical machinery used by the released implementation.


## Control parameterization

The implementation selector fixes the control parameterization as part of
each reproducibility profile. It is not an independent configuration
option.

The `paper` implementation uses

$$dX_t = \left[ b(t,X_t)+\sigma(t)u_\theta(t,X_t) \right]dt +\sigma(t)dW_t,$$

with adjoint-matching target

$$-\sigma(t)\left[\nabla E(X_T)+h(X_T)\right].$$

The `official_repository` implementation uses

$$dX_t = \left[ b(t,X_t)+\sigma(t)^2u_\theta(t,X_t) \right]dt +\sigma(t)dW_t,$$

with adjoint-matching target

$$-\left[\nabla E(X_T)+h(X_T)\right].$$

The former `asbs_control_parameterization` option is intentionally not
supported because it would permit hybrid configurations that correspond
to neither reproducibility profile.

## Repository-style epoch lifecycle

In `official_repository`, each matcher epoch follows the official repository
lifecycle:

1. obtain a fresh source batch;
2. generate fresh endpoint pairs;
3. append the endpoint pairs to persistent replay;
4. rebuild the retained replay dataset;
5. perform `asbs_train_iterations_per_epoch` optimizer updates.

The optional `source_sampler(key, nsamples)` argument allows fresh source
states to be sampled every epoch. Without it, `x_initial` is treated as
an empirical source pool for backward compatibility.

The configured `asbs_train_batch_size` is used unless an explicit
`batch_size` override is supplied. Resolved ASBS configuration is not
mutated during training.

## Repository-style clipping and loss normalization

The optional `asbs_target_clip` setting is restricted to `official_repository`.
It reproduces the official implementation's per-sample terminal
energy-gradient clipping:

$$\widetilde{\nabla E} = \min\left( 1, \dfrac{c}{\lVert \nabla E\rVert_2 + 10^{-6}} \right)\nabla E.$$

The terminal adjoint is then formed as

$$a_T = \widetilde{\nabla E(X_T)} + h(X_T).$$

Thus, only the energy gradient is clipped; the corrector is added
afterward.

The two implementations retain their respective loss normalizations. For residuals
$r_i \in \mathbb{R}^d$, `paper` uses

$$L_{\mathrm{paper}} = \dfrac{1}{2B} \sum_{i=1}^{B} \lVert r_i\rVert_2^2,$$

while `official_repository` follows the official elementwise mean squared error,

$$L_{\mathrm{repo}} = \dfrac{1}{Bd} \sum_{i=1}^{B} \sum_{j=1}^{d} r_{ij}^2.$$

## Initialization blocks and frozen adjoint replay

The official implementation marks every epoch in the first matching block
as an ASBS initialization epoch.

For an initial adjoint block, terminal adjoints omit the corrector:

$$a_T = \nabla E(X_T).$$

For later adjoint blocks, they include the corrector:

$$a_T = \nabla E(X_T) + h(X_T).$$

For an initial corrector block, endpoints are generated with the
uncontrolled reference process throughout the block. Later corrector
blocks use the controlled process.

Repository-style adjoint replay stores the terminal adjoint when each
endpoint pair enters the buffer. Replayed samples therefore retain their
original targets rather than being relabeled using a later corrector.

The public repository loop and repeated paper-stage calls retain stage
progression within a sampler instance. The current sampler has no
sampler-level checkpoint API, so restart persistence for these stage
counters is not claimed.

## Repository-style minibatch traversal

Within each repository epoch, the expanded replay dataset is randomly
shuffled and traversed without replacement. The final minibatch in a
traversal may be smaller than `asbs_train_batch_size`.

If `asbs_train_iterations_per_epoch` requires more updates than one traversal
provides, a new random permutation begins and traversal continues. A
configured batch size larger than the dataset therefore yields the whole
dataset rather than sampling repeated entries.

## Repository-style low-dimensional networks

For low-dimensional `official_repository` experiments, the controller and
corrector both use the official time-dependent Fourier MLP architecture.
The defaults are four layers, 64 channels, GELU activations, trainable
Fourier phases, and frequencies linearly spaced from 0.1 to 100.

Both output layers are initialized with zero weights and biases. The
corrector is evaluated at normalized terminal time \(t=1\), matching the
official repository.

The optional `asbs_model_channels` and `asbs_model_num_layers` settings
override the corresponding architecture dimensions. The
`official_repository` profile currently fixes the model family to the
low-dimensional Fourier MLP. The official particle-system EGNN
architecture is not implemented.

The `paper` implementation continues to use the configured generic RMC controller and
static corrector architectures.

## Time-grid semantics

ASBS follows the established RMC time-discretization convention for both
implementation profiles:

- `h` is the Euler integration step size;
- `T` is the number of integration intervals;
- the terminal time is `h * T`;
- generated paths contain `T + 1` grid points.

Thus,

$$t_k = kh, \qquad k=0,\ldots,T.$$

The `paper` implementation permits any positive terminal time.

The `official_repository` implementation requires normalized time,

$$hT=1.$$

The official ASBS repository commonly specifies the number of function
evaluation grid points rather than the number of intervals. To reproduce
an official run with \(N\) grid points, use

$$T=N-1, \qquad h=\dfrac{1}{N-1}.$$

For example, the official 200-point grid is represented in RMC by
`T=199` and `h=1/199`. No constructor argument is ignored by either
implementation.

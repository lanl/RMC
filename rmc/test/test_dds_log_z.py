"""Discrete path-weight and log-normalizer tests for DDS."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from rmc.modules.dds import DenoisingDiffusionSampler

SIGMA = 0.7


class GaussianTarget:
    def __init__(self, constant):
        self.constant = constant

    def log_target(self, x):
        d = x.shape[-1]
        return (
            -0.5 * jnp.sum(x**2) / SIGMA**2
            - 0.5 * d * jnp.log(2.0 * jnp.pi * SIGMA**2)
            + self.constant
        )


class ConstantControl(nnx.Module):
    def __init__(self, vector):
        self.vector = jnp.asarray(vector)

    def eval(self):
        pass

    def __call__(self, x, t):
        return jnp.broadcast_to(self.vector, x.shape)


def make_sampler(parameterization, constant=0.0, control=(0.0, 0.0), K=3):
    config = {
        "seed": 0,
        "dim": 2,
        "layer_widths": [8, 8],
        "activation_func": nnx.silu,
        "nn_type": "time_concat",
    }
    sampler = DenoisingDiffusionSampler(
        config,
        GaussianTarget(constant),
        sigma=SIGMA,
        K=K,
        beta_schedule=lambda steps: jnp.full((steps,), 0.2),
        control_parameterization=parameterization,
    )
    sampler.nnmodel = ConstantControl(control)
    return sampler


@pytest.mark.parametrize("parameterization", ["f", "u"])
@pytest.mark.parametrize("constant", [0.0, 1.3, -0.8])
def test_zero_control_exact_log_z(parameterization, constant):
    sampler = make_sampler(parameterization, constant=constant)
    assert sampler.log_z_estimate is None

    paths, log_weights = sampler.sample(5, jax.random.PRNGKey(7), return_log_weights=True)

    assert len(paths) == sampler.K + 1
    assert log_weights.shape == (5,)
    np.testing.assert_allclose(log_weights, constant, rtol=0, atol=2e-6)
    np.testing.assert_allclose(sampler.log_z_estimate, constant, rtol=0, atol=2e-6)
    assert sampler.evaluation_counts["sampling"]["density"] == 5


@pytest.mark.parametrize("parameterization", ["f", "u"])
def test_weights_match_explicit_gaussian_transition_ratios(parameterization):
    sampler = make_sampler(parameterization, constant=0.4, control=(0.2, -0.1))
    paths, log_weights = sampler.sample(4, jax.random.PRNGKey(9), return_log_weights=True)

    n = paths[-1].shape[0]
    log_target = jax.vmap(sampler.Dcl.log_target)(paths[-1])
    log_reference = jax.vmap(sampler.ref_process.log_pdf)(paths[-1])
    expected = jnp.reshape(log_target, (n,)) - jnp.reshape(log_reference, (n,))

    for k in range(sampler.K):
        beta = sampler.noise_variance[k]
        a = sampler.retention[k]
        control = sampler.nnmodel(paths[k], 0.0)
        shift = sampler._control_shift(control, beta, sampler.lmbda[k])
        covariance = sampler.sigmaSQ * beta
        residual_ref = paths[k + 1] - a * paths[k]
        residual_control = residual_ref - shift
        expected += (
            jnp.sum(residual_control**2, axis=-1) - jnp.sum(residual_ref**2, axis=-1)
        ) / (2.0 * covariance)

    np.testing.assert_allclose(log_weights, expected, rtol=2e-5, atol=2e-5)


def test_equivalent_control_parameterizations():
    beta = 0.2
    sigma_sq = SIGMA**2
    lam = 1.0 - np.sqrt(1.0 - beta)

    f = np.array([0.3, -0.2])
    u = 2.0 * sigma_sq * lam / beta * f

    sampler_f = make_sampler("f", control=f, K=1)
    sampler_u = make_sampler("u", control=u, K=1)

    key = jax.random.PRNGKey(11)
    paths_f, weights_f = sampler_f.sample(3, key, return_log_weights=True)
    paths_u, weights_u = sampler_u.sample(3, key, return_log_weights=True)

    np.testing.assert_allclose(paths_f[-1], paths_u[-1], atol=1e-6)
    np.testing.assert_allclose(weights_f, weights_u, atol=1e-6)


def test_default_sampling_keeps_previous_estimate_and_counts():
    sampler = make_sampler("u", constant=0.5)

    sampler.sample(3, jax.random.PRNGKey(1), return_log_weights=True)
    previous_estimate = sampler.log_z_estimate

    paths = sampler.sample(2, jax.random.PRNGKey(2))
    assert isinstance(paths, list)
    assert len(paths) == sampler.K + 1
    assert sampler.evaluation_counts["sampling"]["density"] == 3
    np.testing.assert_allclose(sampler.log_z_estimate, previous_estimate)


def test_one_particle_weight_shape():
    sampler = make_sampler("f")
    _, weights = sampler.sample(1, jax.random.PRNGKey(3), return_log_weights=True)
    assert weights.shape == (1,)


class ShiftedGaussianTarget:
    def __init__(self, mean, log_z):
        self.mean = jnp.asarray(mean)
        self.log_z = log_z

    def log_target(self, x):
        d = x.shape[-1]
        residual = x - self.mean
        return (
            -0.5 * jnp.sum(residual**2) / SIGMA**2
            - 0.5 * d * jnp.log(2.0 * jnp.pi * SIGMA**2)
            + self.log_z
        )


@pytest.mark.parametrize("parameterization", ["f", "u"])
def test_shifted_gaussian_log_z_with_nonzero_control(parameterization):
    # Locked numerical protocol:
    # N=8192, seed=2026, K=3, beta=0.2,
    # target mean=(0.3,-0.2), exact log Z=0.4.
    # Acceptance threshold: absolute error <= 0.06.
    sampler = make_sampler(
        parameterization,
        constant=0.0,
        control=(0.2, -0.1),
        K=3,
    )
    sampler.Dcl = ShiftedGaussianTarget(
        mean=(0.3, -0.2),
        log_z=0.4,
    )

    _, log_weights = sampler.sample(
        8192,
        jax.random.PRNGKey(2026),
        return_log_weights=True,
    )

    assert log_weights.shape == (8192,)
    assert np.all(np.isfinite(np.asarray(log_weights)))
    assert abs(float(sampler.log_z_estimate) - 0.4) <= 0.06
    assert sampler.evaluation_counts["sampling"]["density"] == 8192

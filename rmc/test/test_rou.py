import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx

from rmc.modules.rou import ReverseOUSampler


def build_config():
    return {
        "seed": 10,
        "dim": 2,
        "layer_widths": [8, 8],
        "activation_func": nnx.silu,
        "nn_type": "time_embed",
    }


def test_constant_rou_coefficients():
    a0 = 0.7
    sigma0 = 1.2
    h = 0.01
    T = 100

    model = ReverseOUSampler(
        build_config(),
        None,
        h,
        T,
        lambda s: a0 + 0.0 * s,
        lambda s: sigma0 + 0.0 * s,
    )

    terminal_time = h * T

    mean_exact = np.exp(-a0 * terminal_time)
    variance_exact = (
        sigma0**2
        / a0
        * (1.0 - np.exp(-2.0 * a0 * terminal_time))
    )

    np.testing.assert_allclose(
        model.noising_mean[-1],
        mean_exact,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        model.noising_variance[-1],
        variance_exact,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_zero_drift_limit():
    sigma0 = 0.8
    h = 0.01
    T = 100

    model = ReverseOUSampler(
        build_config(),
        None,
        h,
        T,
        lambda s: 0.0 * s,
        lambda s: sigma0 + 0.0 * s,
    )

    np.testing.assert_allclose(
        model.noising_mean[-1],
        1.0,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        model.noising_variance[-1],
        2.0 * sigma0**2 * h * T,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_variance_preserving_rou():
    h = 0.01
    T = 100

    def a_schedule(s):
        return 0.2 + 0.8 * s

    def sigma_schedule(s):
        return jnp.sqrt(a_schedule(s))

    model = ReverseOUSampler(
        build_config(),
        None,
        h,
        T,
        a_schedule,
        sigma_schedule,
    )

    np.testing.assert_allclose(
        model.noising_variance,
        1.0 - model.noising_mean**2,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_conditional_endpoints():
    model = ReverseOUSampler(
        build_config(),
        None,
        0.01,
        100,
        lambda s: 0.5 + 0.0 * s,
        lambda s: 1.0 + 0.0 * s,
    )

    np.testing.assert_allclose(model.cond_mean[-1], 1.0)
    np.testing.assert_allclose(model.cond_variance[-1], 0.0)

    np.testing.assert_allclose(
        model.cond_mean[0],
        model.noising_mean[-1],
    )
    np.testing.assert_allclose(
        model.cond_variance[0],
        model.noising_variance[-1],
    )


def test_conditional_residual_matches_full_drift():
    model = ReverseOUSampler(
        build_config(),
        None,
        0.01,
        100,
        lambda s: 0.3 + 0.4 * s,
        lambda s: 0.7 + 0.2 * s,
    )

    key = jax.random.PRNGKey(10)
    key, key_x, key_eta = jax.random.split(key, 3)

    x_terminal = jax.random.normal(key_x, (8, 2))
    eta = jax.random.normal(key_eta, (8, 2))

    # Exclude k=T because the conditional variance is zero there.
    k = jnp.array([0, 10, 20, 30, 40, 50, 70, 90])

    x = model.eval_conditional_sample(
        x_terminal,
        k,
        eta,
    )

    full_drift = model.eval_conditional_drift(
        x_terminal,
        x,
        k,
    )

    residual = model.eval_conditional_residual(
        eta,
        k,
    )

    a = model.a_gen[k][:, None]

    np.testing.assert_allclose(
        full_drift,
        a * x + residual,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

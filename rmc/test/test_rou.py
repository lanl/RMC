import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx

import rmc.modules.rou as rou_module
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
    variance_exact = sigma0**2 / a0 * (1.0 - np.exp(-2.0 * a0 * terminal_time))

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
        x,
        eta,
        k,
    )

    sigma = model.sigma_gen[k][:, None]

    np.testing.assert_allclose(
        full_drift,
        -(sigma**2) * x + residual,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


class StandardNormalTarget:
    """Minimal derivative-free target used for ROU tests."""

    @staticmethod
    def log_target(x):
        return -0.5 * jnp.sum(x**2)


def test_reference_transition_logpdf():
    a0 = 0.6
    sigma0 = 0.9
    h = 0.05

    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        h,
        8,
        lambda s: a0 + 0.0 * s,
        lambda s: sigma0 + 0.0 * s,
    )

    x_prev = jnp.array(
        [
            [0.2, -0.5],
            [1.1, 0.3],
        ]
    )
    x_next = jnp.array(
        [
            [-0.1, 0.7],
            [0.4, -0.2],
        ]
    )

    logpdf = model.eval_reference_transition_logpdf(
        x_prev,
        x_next,
        2,
    )

    mean_coeff = np.exp(-a0 * h)
    variance = sigma0**2 / a0 * (1.0 - np.exp(-2.0 * a0 * h))

    mean = mean_coeff * np.asarray(x_next)

    expected = -0.5 * (
        2 * np.log(2.0 * np.pi * variance)
        + np.sum(
            (np.asarray(x_prev) - mean) ** 2,
            axis=-1,
        )
        / variance
    )

    np.testing.assert_allclose(
        logpdf,
        expected,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_proposal_transition_logpdf():
    h = 0.02

    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        h,
        10,
        lambda s: 0.4 + 0.0 * s,
        lambda s: 0.8 + 0.0 * s,
    )

    x_prev = jnp.array(
        [
            [0.2, -0.4],
            [0.8, 0.1],
        ]
    )
    x_next = jnp.array(
        [
            [0.3, -0.2],
            [0.6, 0.4],
        ]
    )

    k = 3

    logpdf = model.eval_proposal_transition_logpdf(
        x_prev,
        x_next,
        k,
    )

    drift = model.eval_drift(x_prev, k)
    mean = x_prev + h * drift
    variance = 2.0 * model.sigma_gen[k] ** 2 * h

    expected = -0.5 * (
        model.d * jnp.log(2.0 * jnp.pi * variance)
        + jnp.sum((x_next - mean) ** 2, axis=-1) / variance
    )

    np.testing.assert_allclose(
        logpdf,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_generate_weighted_endpoints():
    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.02,
        10,
        lambda s: 0.5 + 0.0 * s,
        lambda s: 1.0 + 0.0 * s,
    )

    key = jax.random.PRNGKey(100)

    endpoints, log_weights, diagnostics = model.generate_weighted_endpoints(
        64,
        key,
    )

    assert endpoints.shape == (64, 2)
    assert log_weights.shape == (64,)

    assert bool(jnp.all(jnp.isfinite(endpoints)))
    assert bool(jnp.all(jnp.isfinite(log_weights)))

    np.testing.assert_allclose(
        jnp.sum(diagnostics["weights"]),
        1.0,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    assert float(diagnostics["ess"]) >= 1.0
    assert float(diagnostics["ess"]) <= 64.0 + 1.0e-5
    assert float(diagnostics["ess_fraction"]) > 0.0
    assert float(diagnostics["ess_fraction"]) <= 1.0 + 1.0e-6


def test_resample_endpoints():
    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.02,
        5,
        lambda s: 0.5 + 0.0 * s,
        lambda s: 1.0 + 0.0 * s,
    )

    x_terminal = jnp.arange(20, dtype=jnp.float32).reshape(10, 2)

    # Concentrate almost all mass on particle 3.
    log_weights = jnp.full((10,), -100.0)
    log_weights = log_weights.at[3].set(0.0)

    key = jax.random.PRNGKey(10)

    resampled = model.resample_endpoints(
        key,
        x_terminal,
        log_weights,
        32,
    )

    expected = jnp.tile(
        x_terminal[3][None, :],
        (32, 1),
    )

    np.testing.assert_allclose(resampled, expected)


def test_conditional_regression_loss():
    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.01,
        32,
        lambda s: 0.5 + 0.0 * s,
        lambda s: jnp.sqrt(0.5) + 0.0 * s,
    )

    key = jax.random.PRNGKey(123)

    x_terminal = jax.random.normal(
        key,
        (64, 2),
    )

    loss = model.compute_loss(
        model.nnmodel,
        x_terminal,
        jnp.zeros_like(x_terminal),
        key,
    )

    assert bool(jnp.isfinite(loss))
    assert float(loss) >= 0.0


def test_rou_sample():
    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.01,
        16,
        lambda s: 0.5 + 0.0 * s,
        lambda s: jnp.sqrt(0.5) + 0.0 * s,
    )

    key = jax.random.PRNGKey(321)

    xpath = model.sample(
        32,
        key,
    )

    assert len(xpath) == 17

    path = jnp.asarray(xpath)

    assert path.shape == (17, 32, 2)
    assert bool(jnp.all(jnp.isfinite(path)))


def test_vp_stationary_population_residual_is_zero():
    """For a stationary VP OU path, the optimal residual drift is zero."""
    a0 = 0.5

    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.01,
        32,
        lambda s: a0 + 0.0 * s,
        lambda s: jnp.sqrt(a0) + 0.0 * s,
    )

    k = jnp.array([0, 4, 8, 16, 24, 30])

    x = jnp.array(
        [
            [1.0, -0.5],
            [-0.2, 0.7],
            [0.4, 1.2],
            [-1.0, 0.3],
            [0.6, -0.8],
            [0.2, 0.1],
        ]
    )

    variance = model.cond_variance[k][:, None]

    # For the stationary Gaussian VP path,
    #
    #   E[eta | X_t=x] = sqrt(c(t)) x.
    #
    # Therefore E[g_cond | X_t=x] = 0.
    eta_conditional_mean = jnp.sqrt(variance) * x

    residual = model.eval_conditional_residual(
        x,
        eta_conditional_mean,
        k,
    )

    np.testing.assert_allclose(
        residual,
        jnp.zeros_like(x),
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_resample_one_step_reuses_optimizer(tmp_path, monkeypatch):
    config = build_config()
    config.update(
        {
            "opt_type": "ADAM",
            "base_lr": 1.0e-3,
            "opt_grad_max_norm": 10.0,
            "max_samples": 8,
            "nsamples": 4,
            "rou_training_mode": "resample_one_step",
            "rou_resample_size": 4,
            "rou_conditional_samples": 2,
            "rou_outer_iterations": 2,
            "rou_log_every": 10,
            "has_aux": False,
            "root_path": str(tmp_path),
        }
    )

    model = ReverseOUSampler(
        config,
        StandardNormalTarget(),
        0.02,
        4,
        lambda s: 0.5 + 0.0 * s,
        lambda s: jnp.sqrt(0.5) + 0.0 * s,
    )

    probe_x = jnp.array(
        [
            [0.25, -0.5],
            [-0.4, 0.2],
        ]
    )
    probe_t = jnp.full((2, 1), 0.04)

    output_before = np.asarray(model.nnmodel(probe_x, probe_t))

    original_train_step = rou_module.train_step
    optimizer_ids = []

    def tracking_train_step(
        nnmodel,
        criterion,
        optimizer,
        metrics,
        x,
        y,
        has_aux=False,
    ):
        optimizer_ids.append(id(optimizer))
        return original_train_step(
            nnmodel,
            criterion,
            optimizer,
            metrics,
            x,
            y,
            has_aux,
        )

    monkeypatch.setattr(
        rou_module,
        "train_step",
        tracking_train_step,
    )

    history = model.train()

    output_after = np.asarray(model.nnmodel(probe_x, probe_t))

    assert len(history) == 2

    assert all(np.isfinite(entry["loss"]) for entry in history)
    assert all(np.isfinite(entry["ess_fraction"]) for entry in history)

    # Exactly one optimizer update occurs per proposal refresh.
    assert len(optimizer_ids) == 2

    # Both updates use the same NNX optimizer, so Adam state persists.
    assert optimizer_ids[0] == optimizer_ids[1]

    # Training actually updates the neural proposal.
    assert not np.allclose(
        output_before,
        output_after,
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_continuous_conditional_moments_constant_ou():
    a0 = 0.7
    sigma0 = 0.9

    model = ReverseOUSampler(
        build_config(),
        StandardNormalTarget(),
        0.01,
        500,
        lambda s: a0 + 0.0 * s,
        lambda s: sigma0 + 0.0 * s,
    )

    s = jnp.array(
        [1.0e-4, 0.0037, 0.01, 0.137, 1.234, 4.999],
        dtype=jnp.float32,
    )

    mean, variance, a, sigma = model._eval_continuous_noising_moments(s)

    exact_mean = jnp.exp(-a0 * s)
    exact_variance = sigma0**2 / a0 * (1.0 - jnp.exp(-2.0 * a0 * s))

    np.testing.assert_allclose(
        mean,
        exact_mean,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        variance,
        exact_variance,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_allclose(a, a0)
    np.testing.assert_allclose(sigma, sigma0)


def test_continuous_training_dataset_uses_normalized_time():
    config = build_config()
    config.update(
        {
            "rou_conditional_samples": 3,
            "rou_conditional_time_sampling": "continuous",
            "rou_conditional_time_epsilon": 1.0e-4,
            "rou_normalize_time": True,
        }
    )

    model = ReverseOUSampler(
        config,
        StandardNormalTarget(),
        0.01,
        500,
        lambda s: 1.0 + 0.0 * s,
        lambda s: 1.0 + 0.0 * s,
    )

    endpoints = jnp.array(
        [
            [-1.0, -1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    weights = jnp.full((4,), 0.25)

    ds = model._build_weighted_training_dataset(
        endpoints,
        weights,
        jax.random.PRNGKey(123),
    )

    assert ds["input"].shape == (12, 3)
    assert ds["label"].shape == (12, 3)

    nn_time = ds["input"][:, 2]

    assert bool(jnp.all(nn_time >= 0.0))
    assert bool(jnp.all(nn_time <= 1.0))
    assert bool(jnp.all(jnp.isfinite(ds["input"])))
    assert bool(jnp.all(jnp.isfinite(ds["label"])))

    np.testing.assert_allclose(
        model._network_time(jnp.array([0.0, 2.5, 5.0])),
        jnp.array([0.0, 0.5, 1.0]),
    )

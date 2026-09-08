import jax
import jax.numpy as jnp
import numpy as np

from flax import nnx

from rmc.modules.adjoint import AdjointSampler
from rmc.utils.density import BaseLogDensity
from rmc.utils.schedule_diffusion import (
    constant_diffusion_schedule,
    constant_integrated_variance,
    geometric_diffusion_schedule,
    geometric_integrated_variance,
)


class IsotropicGaussianTarget(BaseLogDensity):
    """Gaussian target with analytically known score."""

    def __init__(self, mean, variance):
        self.mean = jnp.asarray(mean)
        self.variance = variance

    def log_target(self, x):
        return -0.5 * jnp.sum((x - self.mean) ** 2) / self.variance


def build_config():
    return {
        "seed": 10,
        "dim": 2,
        "layer_widths": [8, 8],
        "activation_func": nnx.silu,
        "nn_type": "time",
    }


def build_constant_model(
    mean=(0.0, 0.0),
    target_variance=1.0,
    sigma=1.0,
    h=0.01,
    T=100,
):
    target = IsotropicGaussianTarget(
        mean=mean,
        variance=target_variance,
    )

    return AdjointSampler(
        config=build_config(),
        densitycl=target,
        h=h,
        T=T,
        sigma_schedule=lambda t: sigma + 0.0 * t,
        integrated_variance=lambda t: sigma**2 * t,
    )


def test_constant_diffusion_integrated_variance():
    sigma = 0.7
    t = jnp.array([0.0, 0.1, 0.5, 1.0])

    np.testing.assert_allclose(
        constant_diffusion_schedule(t, sigma),
        sigma * np.ones(4),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        constant_integrated_variance(t, sigma),
        sigma**2 * np.asarray(t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_geometric_integrated_variance_endpoints():
    sigma_min = 0.2
    sigma_max = 2.0

    q0 = geometric_integrated_variance(
        0.0,
        sigma_min,
        sigma_max,
    )
    q1 = geometric_integrated_variance(
        1.0,
        sigma_min,
        sigma_max,
    )

    np.testing.assert_allclose(q0, 0.0, atol=1.0e-7)

    np.testing.assert_allclose(
        q1,
        sigma_max**2 - sigma_min**2,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_geometric_integrated_variance_derivative_matches_sigma_squared():
    sigma_min = 0.2
    sigma_max = 2.0
    terminal_time = 2.0

    t = jnp.array([0.2, 0.6, 1.2, 1.7])
    eps = 1.0e-3

    q_plus = geometric_integrated_variance(
        t + eps,
        sigma_min,
        sigma_max,
        terminal_time,
    )
    q_minus = geometric_integrated_variance(
        t - eps,
        sigma_min,
        sigma_max,
        terminal_time,
    )

    derivative = (q_plus - q_minus) / (2.0 * eps)

    sigma = geometric_diffusion_schedule(
        t,
        sigma_min,
        sigma_max,
        terminal_time,
    )

    np.testing.assert_allclose(
        derivative,
        sigma**2,
        rtol=2.0e-3,
        atol=2.0e-4,
    )


def test_terminal_gradient_matched_base_is_zero():
    model = build_constant_model(
        target_variance=1.0,
        sigma=1.0,
    )

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )

    np.testing.assert_allclose(
        model.eval_terminal_gradient(x),
        jnp.zeros_like(x),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_terminal_gradient_shifted_gaussian():
    sigma = 0.8
    h = 0.02
    T = 50

    mean = jnp.array([0.5, -1.0])
    target_variance = 0.4

    model = build_constant_model(
        mean=mean,
        target_variance=target_variance,
        sigma=sigma,
        h=h,
        T=T,
    )

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )

    q_terminal = sigma**2 * h * T

    expected = -x / q_terminal + (x - mean) / target_variance
    actual = model.eval_terminal_gradient(x)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_constant_diffusion_bridge_moments():
    sigma = 0.7

    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    t = jnp.array([0.0, 0.2, 0.5, 0.9, 1.0])

    alpha, variance = model.eval_bridge_moments(t)

    np.testing.assert_allclose(
        alpha,
        t,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        variance,
        sigma**2 * t * (1.0 - t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_sample_base_bridge_matches_reparameterization():
    sigma = 0.7

    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    x_terminal = jnp.array(
        [
            [1.0, -1.0],
            [0.5, 0.25],
            [-0.75, 1.25],
        ]
    )

    t = jnp.array([0.2, 0.5, 0.9])

    noise = jnp.array(
        [
            [0.3, -0.2],
            [-1.0, 0.4],
            [0.1, 0.7],
        ]
    )

    actual = model.sample_base_bridge(
        x_terminal,
        t,
        noise,
    )

    variance = sigma**2 * t * (1.0 - t)

    expected = t[:, None] * x_terminal + jnp.sqrt(variance)[:, None] * noise

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


class ZeroControl(nnx.Module):
    """Control used to test the uncontrolled reference process."""

    def __call__(self, x, t):
        return jnp.zeros_like(x)


def test_eval_drift_is_sigma_times_control():
    sigma = 0.7
    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )
    t = jnp.array([0.2, 0.5, 0.9])

    control = model.eval_control(x, t)
    drift = model.eval_drift(x, t)

    np.testing.assert_allclose(
        drift,
        sigma * control,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_zero_control_rollout_recovers_base_process():
    sigma = 0.7
    h = 0.05
    T = 20
    nsamples = 20000

    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
        h=h,
        T=T,
    )
    model.nnmodel = ZeroControl()

    paths = model.generate_paths(
        nsamples,
        jax.random.PRNGKey(123),
    )

    assert len(paths) == T + 1
    assert paths[0].shape == (nsamples, 2)

    np.testing.assert_allclose(
        paths[0],
        jnp.zeros_like(paths[0]),
        atol=0.0,
    )

    terminal = np.asarray(paths[-1])

    mean = np.mean(terminal, axis=0)
    covariance = np.cov(
        terminal,
        rowvar=False,
        ddof=0,
    )

    expected_variance = sigma**2 * h * T

    np.testing.assert_allclose(
        mean,
        np.zeros(2),
        atol=2.0e-2,
    )
    np.testing.assert_allclose(
        covariance,
        expected_variance * np.eye(2),
        atol=2.0e-2,
    )


def test_generate_endpoints_matches_last_path():
    sigma = 0.7
    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
        h=0.05,
        T=5,
    )
    model.nnmodel = ZeroControl()

    key = jax.random.PRNGKey(42)

    paths = model.generate_paths(16, key)
    endpoints = model.generate_endpoints(16, key)

    np.testing.assert_allclose(
        endpoints,
        paths[-1],
        rtol=0.0,
        atol=0.0,
    )


def test_ram_batch_shapes_and_target():
    sigma = 0.7
    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    endpoints = jnp.array(
        [
            [1.0, -1.0],
            [0.5, 0.25],
            [-0.75, 1.25],
        ]
    )

    terminal_gradients = jnp.array(
        [
            [0.2, -0.4],
            [1.0, 0.5],
            [-0.3, 0.7],
        ]
    )

    batch = model.build_ram_batch(
        jax.random.PRNGKey(5),
        endpoints,
        terminal_gradients,
    )

    assert batch["input"].shape == (3, 3)
    assert batch["label"].shape == (3, 2)

    times = batch["input"][:, -1]

    assert jnp.all(times >= 0.0)
    assert jnp.all(times < model.TT)

    np.testing.assert_allclose(
        batch["label"],
        -sigma * terminal_gradients,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_ram_batch_subsamples_endpoint_gradient_pairs():
    sigma = 0.7
    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    endpoints = jnp.arange(
        20,
        dtype=jnp.float32,
    ).reshape(10, 2)

    terminal_gradients = endpoints + 100.0

    batch = model.build_ram_batch(
        jax.random.PRNGKey(9),
        endpoints,
        terminal_gradients,
        batch_size=6,
    )

    assert batch["input"].shape == (6, 3)
    assert batch["label"].shape == (6, 2)

    recovered_gradients = -batch["label"] / sigma

    for gradient in np.asarray(recovered_gradients):
        assert any(np.allclose(gradient, candidate) for candidate in np.asarray(terminal_gradients))


def test_ram_loss_is_vector_squared_error():
    sigma = 0.7
    model = build_constant_model(
        target_variance=sigma**2,
        sigma=sigma,
    )

    zero_control = ZeroControl()

    input = jnp.array(
        [
            [0.2, -0.5, 0.1],
            [1.0, 0.4, 0.7],
        ]
    )

    labels = jnp.array(
        [
            [1.0, -2.0],
            [0.5, 0.25],
        ]
    )

    expected = 0.5 * jnp.mean(jnp.sum(labels**2, axis=-1))

    actual = model.compute_ram_loss(
        zero_control,
        input,
        labels,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_replay_buffer_accumulates_and_preserves_pairs():
    from rmc.modules.adjoint import _ReplayBuffer

    buffer = _ReplayBuffer(dim=2)

    assert len(buffer) == 0

    endpoints_1 = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    gradients_1 = endpoints_1 + 100.0

    endpoints_2 = jnp.array(
        [
            [5.0, 6.0],
        ]
    )
    gradients_2 = endpoints_2 + 100.0

    buffer.add(
        endpoints_1,
        gradients_1,
    )
    buffer.add(
        endpoints_2,
        gradients_2,
    )

    assert len(buffer) == 3

    np.testing.assert_allclose(
        buffer.endpoints,
        jnp.concatenate(
            [endpoints_1, endpoints_2],
            axis=0,
        ),
    )
    np.testing.assert_allclose(
        buffer.terminal_gradients,
        jnp.concatenate(
            [gradients_1, gradients_2],
            axis=0,
        ),
    )

    endpoints, gradients = buffer.sample(
        jax.random.PRNGKey(3),
        batch_size=20,
    )

    np.testing.assert_allclose(
        gradients,
        endpoints + 100.0,
    )


def test_replay_buffer_rejects_empty_sample():
    import pytest

    from rmc.modules.adjoint import _ReplayBuffer

    buffer = _ReplayBuffer(dim=2)

    with pytest.raises(
        ValueError,
        match="empty replay buffer",
    ):
        buffer.sample(
            jax.random.PRNGKey(1),
            batch_size=2,
        )


def test_naive_training_loop_matches_algorithm_one(
    monkeypatch,
):
    import rmc.modules.adjoint as adjoint_module

    config = build_config()
    config.update(
        {
            "opt_type": "ADAM",
            "base_lr": 1.0e-3,
            "opt_grad_max_norm": 1.0e20,
            "adjoint_repo_features": False,
            "adjoint_outer_iterations": 2,
            "adjoint_outer_samples": 8,
            "adjoint_inner_steps": 3,
            "adjoint_batch_size": 4,
            "adjoint_log_every": 100,
        }
    )

    target = IsotropicGaussianTarget(
        mean=jnp.zeros(2),
        variance=1.0,
    )

    model = AdjointSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=10,
        sigma_schedule=lambda t: 1.0 + 0.0 * t,
        integrated_variance=lambda t: t,
    )

    counters = {
        "rollouts": 0,
        "gradient_points": 0,
        "train_steps": 0,
        "optimizer_ids": [],
    }

    original_generate_endpoints = model.generate_endpoints
    original_eval_terminal_gradient = model.eval_terminal_gradient

    def counted_generate_endpoints(
        nsamples,
        subkey,
    ):
        counters["rollouts"] += 1
        return original_generate_endpoints(
            nsamples,
            subkey,
        )

    def counted_terminal_gradient(x):
        counters["gradient_points"] += x.shape[0]
        return original_eval_terminal_gradient(x)

    def fake_train_step(
        nnmodel,
        criterion,
        optimizer,
        metrics,
        input,
        labels,
        has_aux,
    ):
        counters["train_steps"] += 1
        counters["optimizer_ids"].append(id(optimizer))

        return criterion(
            nnmodel,
            input,
            labels,
        )

    monkeypatch.setattr(
        model,
        "generate_endpoints",
        counted_generate_endpoints,
    )
    monkeypatch.setattr(
        model,
        "eval_terminal_gradient",
        counted_terminal_gradient,
    )
    monkeypatch.setattr(
        adjoint_module,
        "train_step",
        fake_train_step,
    )

    history = model.train()

    assert counters["rollouts"] == 2
    assert counters["gradient_points"] == 16
    assert counters["train_steps"] == 6

    assert len(set(counters["optimizer_ids"])) == 1

    assert len(history) == 2

    assert history[0]["buffer_size"] == 8
    assert history[1]["buffer_size"] == 16

    assert np.isfinite(history[0]["loss"])
    assert np.isfinite(history[1]["loss"])


def test_replay_buffer_capacity_is_fifo():
    from rmc.modules.adjoint import _ReplayBuffer

    buffer = _ReplayBuffer(
        dim=2,
        capacity=3,
    )

    first = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    second = jnp.array(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    buffer.add(
        first,
        first + 100.0,
    )
    buffer.add(
        second,
        second + 100.0,
    )

    expected = jnp.array(
        [
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    assert len(buffer) == 3

    np.testing.assert_allclose(
        buffer.endpoints,
        expected,
    )
    np.testing.assert_allclose(
        buffer.terminal_gradients,
        expected + 100.0,
    )


def test_repo_feature_defaults_and_overrides():
    config = build_config()
    config["adjoint_repo_features"] = True

    target = IsotropicGaussianTarget(
        mean=jnp.zeros(2),
        variance=1.0,
    )

    model = AdjointSampler(
        config=config,
        densitycl=target,
        h=0.02,
        T=50,
    )

    options = model._resolve_training_options()

    assert options["inner_steps"] == 100
    assert options["replay_capacity"] == 1000
    assert options["init_base_samples"] == 1024
    assert options["target_clip"] == 150.0
    assert options["time_discretization"] == "ql"

    np.testing.assert_allclose(
        model.base_terminal_variance,
        1.0 - 1.0e-6,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    times = model._build_time_grid()

    assert times.shape == (51,)
    np.testing.assert_allclose(times[0], 0.0)
    np.testing.assert_allclose(times[-1], 1.0)

    assert jnp.all(jnp.diff(times) > 0.0)

    config["adjoint_replay_capacity"] = 7
    config["adjoint_init_base_samples"] = 11
    config["adjoint_target_clip"] = None
    config["adjoint_time_discretization"] = "uniform"
    config["adjoint_inner_steps"] = 3

    options = model._resolve_training_options()

    assert options["inner_steps"] == 3
    assert options["replay_capacity"] == 7
    assert options["init_base_samples"] == 11
    assert options["target_clip"] is None
    assert options["time_discretization"] == "uniform"


def test_terminal_gradient_norm_clipping():
    model = build_constant_model()

    gradients = jnp.array(
        [
            [3.0, 4.0],
            [0.1, 0.0],
        ]
    )

    clipped, fraction = model._clip_terminal_gradients(
        gradients,
        max_norm=1.0,
    )

    norms = jnp.linalg.norm(
        clipped,
        axis=-1,
    )

    np.testing.assert_allclose(
        norms,
        jnp.array([1.0, 0.1]),
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    np.testing.assert_allclose(
        fraction,
        0.5,
    )


def test_repo_training_initializes_and_bounds_replay(
    monkeypatch,
):
    import rmc.modules.adjoint as adjoint_module

    config = build_config()
    config.update(
        {
            "opt_type": "ADAM",
            "base_lr": 1.0e-3,
            "opt_grad_max_norm": 1.0e20,
            "adjoint_repo_features": True,
            "adjoint_outer_iterations": 2,
            "adjoint_outer_samples": 4,
            "adjoint_inner_steps": 2,
            "adjoint_batch_size": 3,
            "adjoint_init_base_samples": 5,
            "adjoint_replay_capacity": 6,
            "adjoint_target_clip": 1.0,
            "adjoint_time_discretization": "uniform",
            "adjoint_log_every": 100,
        }
    )

    target = IsotropicGaussianTarget(
        mean=jnp.zeros(2),
        variance=1.0,
    )

    model = AdjointSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=10,
        sigma_schedule=lambda t: 1.0 + 0.0 * t,
        integrated_variance=lambda t: t,
    )

    counters = {
        "base_calls": 0,
        "rollouts": 0,
        "gradient_points": 0,
        "train_steps": 0,
    }

    def fake_base_endpoints(
        nsamples,
        subkey,
    ):
        counters["base_calls"] += 1
        return jnp.zeros((nsamples, 2))

    def fake_generate_endpoints(
        nsamples,
        subkey,
    ):
        counters["rollouts"] += 1
        return jnp.ones((nsamples, 2))

    def fake_terminal_gradient(x):
        counters["gradient_points"] += x.shape[0]
        return 10.0 * jnp.ones_like(x)

    def fake_train_step(
        nnmodel,
        criterion,
        optimizer,
        metrics,
        input,
        labels,
        has_aux,
    ):
        counters["train_steps"] += 1
        return criterion(
            nnmodel,
            input,
            labels,
        )

    monkeypatch.setattr(
        model,
        "sample_base_endpoints",
        fake_base_endpoints,
    )
    monkeypatch.setattr(
        model,
        "generate_endpoints",
        fake_generate_endpoints,
    )
    monkeypatch.setattr(
        model,
        "eval_terminal_gradient",
        fake_terminal_gradient,
    )
    monkeypatch.setattr(
        adjoint_module,
        "train_step",
        fake_train_step,
    )

    history = model.train()

    assert counters["base_calls"] == 1
    assert counters["rollouts"] == 2
    assert counters["gradient_points"] == 13
    assert counters["train_steps"] == 4

    assert len(history) == 2
    assert history[0]["buffer_size"] == 6
    assert history[1]["buffer_size"] == 6

    np.testing.assert_allclose(
        history[0]["terminal_gradient_clip_fraction"],
        1.0,
    )
    np.testing.assert_allclose(
        history[1]["terminal_gradient_clip_fraction"],
        1.0,
    )

import jax
import jax.numpy as jnp

import numpy as np
from flax import nnx

from rmc.modules.schrodinger_bridge import SchrodingerBridgeSampler
from rmc.utils.density import BaseLogDensity


class IsotropicGaussianTarget(BaseLogDensity):
    """Gaussian Boltzmann target with analytically known energy gradient."""

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
        "asbs_mode": "paper",
        "nn_type": "time",
        "opt_type": "ADAM",
        "base_lr": 1.0e-3,
        "opt_grad_max_norm": 10.0,
    }


def build_sampler(
    sigma=0.7,
    h=0.01,
    T=100,
    mean=(0.0, 0.0),
    target_variance=1.0,
):
    config = build_config()
    target = IsotropicGaussianTarget(
        mean=mean,
        variance=target_variance,
    )

    return SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=h,
        T=T,
        sigma_schedule=lambda t: sigma + 0.0 * t,
        integrated_variance=lambda t: sigma**2 * t,
    )


def test_asbs_replay_buffer_accumulates_batches():
    from rmc.modules.schrodinger_bridge import _ASBSReplayBuffer

    buffer = _ASBSReplayBuffer(capacity=5)

    buffer.add(
        {
            "x0": jnp.arange(6.0).reshape(3, 2),
            "x1": jnp.arange(6.0, 12.0).reshape(3, 2),
        }
    )

    assert len(buffer) == 3

    dataset = buffer.build_dataset()

    np.testing.assert_allclose(
        dataset["x0"],
        jnp.arange(6.0).reshape(3, 2),
    )
    np.testing.assert_allclose(
        dataset["x1"],
        jnp.arange(6.0, 12.0).reshape(3, 2),
    )


def test_asbs_replay_buffer_is_fifo():
    from rmc.modules.schrodinger_bridge import _ASBSReplayBuffer

    buffer = _ASBSReplayBuffer(capacity=3)

    buffer.add(
        {
            "x0": jnp.array(
                [
                    [0.0],
                    [1.0],
                ]
            )
        }
    )
    buffer.add(
        {
            "x0": jnp.array(
                [
                    [2.0],
                    [3.0],
                ]
            )
        }
    )

    dataset = buffer.build_dataset()

    np.testing.assert_allclose(
        dataset["x0"],
        jnp.array(
            [
                [1.0],
                [2.0],
                [3.0],
            ]
        ),
    )


def test_asbs_replay_buffer_duplicates_dataset():
    from rmc.modules.schrodinger_bridge import _ASBSReplayBuffer

    buffer = _ASBSReplayBuffer(capacity=4)

    buffer.add(
        {
            "x0": jnp.array(
                [
                    [0.0],
                    [1.0],
                ]
            )
        }
    )

    dataset = buffer.build_dataset(duplicates=3)

    np.testing.assert_allclose(
        dataset["x0"],
        jnp.array(
            [
                [0.0],
                [1.0],
                [0.0],
                [1.0],
                [0.0],
                [1.0],
            ]
        ),
    )


def test_asbs_replay_buffer_rejects_empty_buffer():
    from rmc.modules.schrodinger_bridge import _ASBSReplayBuffer

    buffer = _ASBSReplayBuffer(capacity=4)

    try:
        buffer.build_dataset()
    except ValueError as exc:
        assert "empty replay buffer" in str(exc)
    else:
        raise AssertionError("Expected empty replay buffer to fail")


def test_paper_mode_is_default_and_resolved():
    sampler = build_sampler()

    assert sampler.asbs_options["mode"] == "paper"


def test_unsupported_asbs_mode_fails_early():
    config = build_config()
    config["asbs_mode"] = "invalid"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    try:
        SchrodingerBridgeSampler(
            config=config,
            densitycl=target,
            h=0.01,
            T=100,
        )
    except ValueError as exc:
        assert "Unsupported ASBS mode" in str(exc)
    else:
        raise AssertionError("Expected invalid ASBS mode to fail")


def test_paper_repo_mode_resolves_replay_options():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_replay_capacity"] = 7
    config["asbs_replay_duplicates"] = 3

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.01,
        T=100,
    )

    assert sampler.asbs_options["mode"] == "paper_repo"
    assert sampler.asbs_options["replay_capacity"] == 7
    assert sampler.asbs_options["replay_duplicates"] == 3


def test_paper_repo_adjoint_stage_uses_persistent_replay():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_replay_capacity"] = 6
    config["asbs_replay_duplicates"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    first = sampler.train_repo_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    second = sampler.train_repo_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(1),
        inner_steps=1,
    )

    assert first["buffer_size"] == 3
    assert second["buffer_size"] == 6
    assert bool(jnp.isfinite(first["adjoint_loss"][0]))
    assert bool(jnp.isfinite(second["adjoint_loss"][0]))


def test_paper_repo_corrector_stage_uses_persistent_replay():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_replay_capacity"] = 6
    config["asbs_replay_duplicates"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    first = sampler.train_repo_corrector_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    second = sampler.train_repo_corrector_stage(
        x_initial,
        jax.random.PRNGKey(1),
        inner_steps=1,
    )

    assert first["buffer_size"] == 3
    assert second["buffer_size"] == 6
    assert bool(jnp.isfinite(first["corrector_loss"][0]))
    assert bool(jnp.isfinite(second["corrector_loss"][0]))


def test_reference_endpoint_generation_has_correct_variance():
    sampler = build_sampler(
        sigma=0.5,
        h=0.1,
        T=10,
    )

    x_initial = jnp.zeros((4000, 2))

    endpoints = sampler.generate_reference_endpoints(
        x_initial.shape[0],
        jax.random.PRNGKey(0),
        x_initial=x_initial,
    )

    empirical_second_moment = jnp.mean(endpoints**2)
    expected_second_moment = 0.5**2 * 1.0

    np.testing.assert_allclose(
        empirical_second_moment,
        expected_second_moment,
        rtol=0.08,
        atol=0.01,
    )


def test_repo_optimizers_are_persistent():
    config = build_config()
    config["asbs_mode"] = "paper_repo"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    sampler._initialize_repo_state()

    controller_optimizer_a, corrector_optimizer_a = sampler._get_repo_optimizers()
    controller_optimizer_b, corrector_optimizer_b = sampler._get_repo_optimizers()

    assert controller_optimizer_a is controller_optimizer_b
    assert corrector_optimizer_a is corrector_optimizer_b


def test_paper_repo_outer_loop_alternates_stages():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_replay_capacity"] = 8

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    history = sampler.train_repo(
        x_initial,
        jax.random.PRNGKey(0),
        outer_stages=3,
        adjoint_steps=1,
        corrector_steps=1,
    )

    assert history["stage"] == [
        "adjoint",
        "corrector",
        "adjoint",
    ]
    assert len(history["adjoint_loss"]) == 3
    assert len(history["corrector_loss"]) == 3
    assert history["adjoint_buffer_size"][-1] > 0
    assert history["corrector_buffer_size"][-1] > 0


def test_paper_repo_init_stage_can_start_with_corrector():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_init_stage"] = "corrector"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    history = sampler.train_repo(
        x_initial,
        jax.random.PRNGKey(0),
        outer_stages=3,
        adjoint_steps=1,
        corrector_steps=1,
    )

    assert history["stage"] == [
        "corrector",
        "adjoint",
        "corrector",
    ]


def test_paper_repo_explicit_one_epoch_is_not_overridden():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2
    config["asbs_train_itr_per_epoch"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    history = sampler.train_repo(
        x_initial,
        jax.random.PRNGKey(0),
        outer_stages=1,
        adjoint_steps=1,
    )

    assert history["stage"] == ["adjoint"]
    assert len(history["adjoint_loss"]) == 1


def test_paper_repo_none_uses_configured_stage_counts():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    assert sampler.asbs_options["adjoint_steps"] == 3
    assert sampler.asbs_options["corrector_steps"] == 2


def test_paper_repo_stage_counts_are_resolved():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    assert sampler.asbs_options["adjoint_steps"] == 3
    assert sampler.asbs_options["corrector_steps"] == 2


def test_repo_endpoint_generation_supports_resampling_chunks():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_resample_size"] = 7
    config["asbs_resample_batch_size"] = 3

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    endpoints = sampler._generate_repo_endpoints(
        x_initial,
        jax.random.PRNGKey(0),
        resample_size=7,
        resample_batch_size=3,
    )

    assert endpoints.shape == (7, 2)
    assert bool(jnp.all(jnp.isfinite(endpoints)))


def test_repo_endpoint_generation_is_invariant_to_chunk_size_for_same_seed():
    config = build_config()
    config["asbs_mode"] = "paper_repo"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
            [0.3, 0.1],
        ]
    )

    key = jax.random.PRNGKey(123)

    full = sampler._generate_repo_endpoints(
        x_initial,
        key,
        resample_size=4,
        resample_batch_size=4,
    )
    chunked = sampler._generate_repo_endpoints(
        x_initial,
        key,
        resample_size=4,
        resample_batch_size=2,
    )

    assert full.shape == chunked.shape
    assert bool(jnp.all(jnp.isfinite(full)))
    assert bool(jnp.all(jnp.isfinite(chunked)))


def test_paper_repo_training_iterations_are_resolved():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_train_itr_per_epoch"] = 37

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    assert sampler.asbs_options["train_iterations"] == 37


def test_paper_repo_resampling_options_are_resolved():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_resample_size"] = 200
    config["asbs_resample_batch_size"] = 50
    config["asbs_train_batch_size"] = 25

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    assert sampler.asbs_options["resample_size"] == 200
    assert sampler.asbs_options["resample_batch_size"] == 50
    assert sampler.asbs_options["train_batch_size"] == 25


def test_paper_repo_resampling_options_must_be_positive():
    for name in (
        "asbs_resample_size",
        "asbs_resample_batch_size",
        "asbs_train_batch_size",
    ):
        config = build_config()
        config["asbs_mode"] = "paper_repo"
        config[name] = 0

        target = IsotropicGaussianTarget(
            mean=(0.0, 0.0),
            variance=1.0,
        )

        try:
            SchrodingerBridgeSampler(
                config=config,
                densitycl=target,
                h=0.1,
                T=5,
            )
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f"Expected {name}=0 to fail")


def test_paper_repo_replay_capacity_is_enforced():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_replay_capacity"] = 4

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )

    x_initial = jnp.zeros((3, 2))

    sampler.train_repo_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    result = sampler.train_repo_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(1),
        inner_steps=1,
    )

    assert result["buffer_size"] == 4


def test_geometric_reference_variance_matches_closed_form():
    sigma_min = 1.0e-3
    sigma_max = 2.0
    terminal_time = 1.0

    config = build_config()
    config["asbs_diffusion_schedule"] = "geometric"
    config["asbs_sigma_min"] = sigma_min
    config["asbs_sigma_max"] = sigma_max

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=10,
    )

    sigma = sampler.eval_sigma
    variance = sampler.eval_integrated_variance

    t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    ratio = sigma_max / sigma_min
    expected_sigma = (
        sigma_min
        * ratio ** (1.0 - t / terminal_time)
        * jnp.sqrt(2.0 * jnp.log(ratio) / terminal_time)
    )
    expected_variance = sigma_max**2 * (1.0 - ratio ** (-2.0 * t / terminal_time))

    np.testing.assert_allclose(
        sigma(t),
        expected_sigma,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        variance(t),
        expected_variance,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_paper_repo_accepts_geometric_reference_process():
    config = build_config()
    config["asbs_mode"] = "paper_repo"
    config["asbs_diffusion_schedule"] = "geometric"
    config["asbs_sigma_min"] = 1.0e-3
    config["asbs_sigma_max"] = 2.0

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.01,
        T=100,
    )

    assert sampler.asbs_options["mode"] == "paper_repo"
    assert sampler.asbs_options["diffusion_schedule"] == "geometric"

    t = jnp.array([0.0, sampler.TT / 2.0, sampler.TT])

    sigma = sampler.eval_sigma(t)
    variance = sampler.eval_integrated_variance(t)

    assert bool(jnp.all(sigma > 0.0))
    assert bool(jnp.isclose(variance[0], 0.0))
    assert bool(
        jnp.isclose(
            variance[-1],
            sampler.asbs_options["sigma_max"] ** 2,
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    )


def test_constant_reference_variance():
    sigma = 0.7
    sampler = build_sampler(sigma=sigma)

    t = jnp.array([0.0, 0.2, 0.5, 1.0])

    np.testing.assert_allclose(
        sampler.eval_sigma(t),
        sigma * jnp.ones_like(t),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        sampler.eval_integrated_variance(t),
        sigma**2 * t,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_energy_gradient_matches_gaussian_target():
    mean = jnp.array([0.5, -1.0])
    variance = 0.4
    sampler = build_sampler(
        mean=mean,
        target_variance=variance,
    )

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )

    expected = (x - mean) / variance
    actual = sampler.eval_energy_gradient(x)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_corrector_target_matches_reference_transition_score():
    sigma = 0.7
    sampler = build_sampler(sigma=sigma, h=0.01, T=100)

    x_initial = jnp.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
            [-0.5, 0.25],
        ]
    )
    x_terminal = jnp.array(
        [
            [1.0, 3.0],
            [-2.0, 1.0],
            [0.75, -1.25],
        ]
    )

    q_terminal = sigma**2 * 1.0
    expected = (x_initial - x_terminal) / q_terminal

    actual = sampler.eval_corrector_target(
        x_initial,
        x_terminal,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_corrector_target_rejects_mismatched_shapes():
    sampler = build_sampler()

    x_initial = jnp.zeros((3, 2))
    x_terminal = jnp.zeros((4, 2))

    try:
        sampler.eval_corrector_target(
            x_initial,
            x_terminal,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched endpoint shapes")


def test_corrector_is_zero_initialized():
    sampler = build_sampler()

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )

    np.testing.assert_allclose(
        sampler.eval_corrector(x),
        jnp.zeros_like(x),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_adjoint_batch_samples_interior_times():
    sampler = build_sampler(
        h=0.1,
        T=10,
    )

    x_initial = jnp.zeros((100, 2))
    x_terminal = jnp.ones((100, 2))

    batch = sampler.build_adjoint_batch(
        jax.random.PRNGKey(0),
        x_initial,
        x_terminal,
    )

    times = batch["input"][:, -1]

    assert bool(jnp.all(times > 0.0))
    assert bool(jnp.all(times < sampler.TT))


def test_first_stage_terminal_adjoint_uses_zero_corrector():
    mean = jnp.array([0.5, -1.0])
    variance = 0.4
    sampler = build_sampler(
        mean=mean,
        target_variance=variance,
    )

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
        ]
    )

    expected = (x - mean) / variance

    np.testing.assert_allclose(
        sampler.eval_terminal_adjoint(
            x,
            include_corrector=False,
        ),
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        sampler.eval_adjoint_target(
            x,
            include_corrector=False,
        ),
        -expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_terminal_adjoint_includes_current_corrector():
    mean = jnp.array([0.5, -1.0])
    variance = 0.4
    sampler = build_sampler(
        mean=mean,
        target_variance=variance,
    )

    corrector = lambda x: 0.25 * jnp.ones_like(x)
    sampler.corrector = corrector

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
        ]
    )

    energy_gradient = (x - mean) / variance
    expected_adjoint = energy_gradient + 0.25
    expected_target = -expected_adjoint

    np.testing.assert_allclose(
        sampler.eval_terminal_adjoint(x),
        expected_adjoint,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        sampler.eval_adjoint_target(x),
        expected_target,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_constant_reference_bridge_moments():
    sigma = 0.7
    sampler = build_sampler(sigma=sigma)

    x_initial = jnp.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
        ]
    )
    x_terminal = jnp.array(
        [
            [1.0, 3.0],
            [-2.0, 1.0],
        ]
    )

    t = jnp.array([0.2, 0.5])

    mean, variance = sampler.eval_bridge_moments(
        x_initial,
        x_terminal,
        t,
    )

    expected_mean = x_initial + t[:, None] * (x_terminal - x_initial)
    expected_variance = sigma**2 * t * (1.0 - t)

    np.testing.assert_allclose(
        mean,
        expected_mean,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        variance,
        expected_variance,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_sample_reference_bridge_matches_reparameterization():
    sigma = 0.7
    sampler = build_sampler(sigma=sigma)

    x_initial = jnp.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
        ]
    )
    x_terminal = jnp.array(
        [
            [1.0, 3.0],
            [-2.0, 1.0],
        ]
    )

    t = jnp.array([0.2, 0.5])

    noise = jnp.array(
        [
            [0.3, -0.2],
            [-1.0, 0.4],
        ]
    )

    actual = sampler.sample_reference_bridge(
        x_initial,
        x_terminal,
        t,
        noise,
    )

    expected_mean = x_initial + t[:, None] * (x_terminal - x_initial)
    variance = sigma**2 * t * (1.0 - t)
    expected = expected_mean + jnp.sqrt(variance)[:, None] * noise

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_reference_bridge_score():
    sampler = build_sampler(sigma=0.7)

    x_initial = jnp.array([[0.0, 1.0]])
    x_terminal = jnp.array([[1.0, 3.0]])
    t = jnp.array([0.5])

    mean, variance = sampler.eval_bridge_moments(
        x_initial,
        x_terminal,
        t,
    )

    x = mean + jnp.array([[0.35, -0.20]])

    actual = sampler.eval_reference_bridge_score(
        x,
        x_initial,
        x_terminal,
        t,
    )

    expected = -(x - mean) / variance[:, None]

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_adjoint_loss_matches_paper_target():
    sampler = build_sampler()

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
        ]
    )
    t = jnp.array([0.2, 0.5])

    labels = jnp.array(
        [
            [0.7, -0.2],
            [-0.4, 0.9],
        ]
    )

    class ZeroController(nnx.Module):
        """Zero controller for loss verification."""

        def __call__(self, x, t):
            del t
            return jnp.zeros_like(x)

    model = ZeroController()

    actual = sampler.compute_adjoint_loss(
        model,
        jnp.concatenate([x, t[:, None]], axis=-1),
        labels,
    )

    expected = 0.5 * jnp.mean(jnp.sum(labels**2, axis=-1))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_corrector_loss_matches_paper_target():
    sampler = build_sampler()

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
        ]
    )

    labels = jnp.array(
        [
            [0.7, -0.2],
            [-0.4, 0.9],
        ]
    )

    class ZeroCorrector(nnx.Module):
        """Zero corrector for loss verification."""

        def __call__(self, x):
            return jnp.zeros_like(x)

    model = ZeroCorrector()

    actual = sampler.compute_corrector_loss(
        model,
        x,
        labels,
    )

    expected = 0.5 * jnp.mean(jnp.sum(labels**2, axis=-1))

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_adjoint_batch_has_expected_shapes():
    sampler = build_sampler()

    x_initial = jnp.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
            [-0.5, 0.25],
            [0.75, 1.25],
        ]
    )
    x_terminal = jnp.array(
        [
            [1.0, 3.0],
            [-2.0, 1.0],
            [0.75, -1.25],
            [-1.0, 0.5],
        ]
    )

    batch = sampler.build_adjoint_batch(
        jax.random.PRNGKey(0),
        x_initial,
        x_terminal,
    )

    assert batch["input"].shape == (4, 3)
    assert batch["label"].shape == (4, 2)


def test_corrector_batch_has_expected_shapes():
    sampler = build_sampler()

    x_initial = jnp.array(
        [
            [0.0, 1.0],
            [2.0, -1.0],
            [-0.5, 0.25],
            [0.75, 1.25],
        ]
    )
    x_terminal = jnp.array(
        [
            [1.0, 3.0],
            [-2.0, 1.0],
            [0.75, -1.25],
            [-1.0, 0.5],
        ]
    )

    batch = sampler.build_corrector_batch(
        x_initial,
        x_terminal,
    )

    assert batch["input"].shape == (4, 2)
    assert batch["label"].shape == (4, 2)


def test_one_paper_asbs_stage_runs():
    sampler = build_sampler(
        sigma=0.5,
        h=0.1,
        T=10,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
            [0.3, 0.1],
        ]
    )

    history = sampler.train_one_stage(
        x_initial=x_initial,
        subkey=jax.random.PRNGKey(123),
        adjoint_steps=1,
        corrector_steps=1,
    )

    assert history["adjoint_loss"].shape == (1,)
    assert history["corrector_loss"].shape == (1,)
    assert bool(jnp.isfinite(history["adjoint_loss"][0]))
    assert bool(jnp.isfinite(history["corrector_loss"][0]))


def test_controller_and_corrector_interfaces():
    sampler = build_sampler()

    x = jnp.array(
        [
            [0.2, -0.5],
            [1.0, 0.4],
            [-0.7, 1.3],
        ]
    )
    t = jnp.array([0.2, 0.5, 0.9])

    control = sampler.eval_control(x, t)
    corrector = sampler.eval_corrector(x)

    assert control.shape == x.shape
    assert corrector.shape == x.shape

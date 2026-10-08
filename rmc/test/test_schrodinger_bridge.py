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
        "asbs_implementation": "paper",
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


def test_paper_implementation_is_explicitly_resolved():
    sampler = build_sampler()

    assert sampler.asbs_options["implementation"] == "paper"


def test_asbs_implementation_must_be_explicit():
    config = build_config()
    del config["asbs_implementation"]

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
        assert "asbs_implementation must be explicitly set" in str(exc)
    else:
        raise AssertionError("Expected a missing implementation selector to fail")


def test_unsupported_asbs_implementation_fails_early():
    config = build_config()
    config["asbs_implementation"] = "invalid"

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
        assert "Unsupported ASBS implementation" in str(exc)
    else:
        raise AssertionError("Expected invalid ASBS mode to fail")


def test_official_repository_implementation_resolves_replay_options():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
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

    assert sampler.asbs_options["implementation"] == "official_repository"
    assert sampler.asbs_options["replay_capacity"] == 7
    assert sampler.asbs_options["replay_duplicates"] == 3


def test_official_repository_adjoint_stage_uses_persistent_replay():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 6
    config["asbs_replay_duplicates"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    first = sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    second = sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(1),
        inner_steps=1,
    )

    assert first["buffer_size"] == 3
    assert second["buffer_size"] == 6
    assert bool(jnp.isfinite(first["adjoint_loss"][0]))
    assert bool(jnp.isfinite(second["adjoint_loss"][0]))


def test_official_repository_corrector_stage_uses_persistent_replay():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 6
    config["asbs_replay_duplicates"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    first = sampler._train_official_repository_corrector_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    second = sampler._train_official_repository_corrector_stage(
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
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    sampler._initialize_training_state()

    controller_optimizer_a, corrector_optimizer_a = sampler._get_stage_optimizers()
    controller_optimizer_b, corrector_optimizer_b = sampler._get_stage_optimizers()

    assert controller_optimizer_a is controller_optimizer_b
    assert corrector_optimizer_a is corrector_optimizer_b


def test_official_repository_outer_loop_alternates_stages():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 8

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    history = sampler._train_official_repository_stages(
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


def test_official_repository_init_stage_can_start_with_corrector():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_init_stage"] = "corrector"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
        ]
    )

    history = sampler._train_official_repository_stages(
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


def test_official_repository_explicit_one_epoch_is_not_overridden():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2
    config["asbs_train_iterations_per_epoch"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    history = sampler._train_official_repository_stages(
        x_initial,
        jax.random.PRNGKey(0),
        outer_stages=1,
        adjoint_steps=1,
    )

    assert history["stage"] == ["adjoint"]
    assert len(history["adjoint_loss"]) == 1


def test_official_repository_none_uses_configured_stage_counts():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    assert sampler.asbs_options["adjoint_steps"] == 3
    assert sampler.asbs_options["corrector_steps"] == 2


def test_official_repository_stage_counts_are_resolved():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_adjoint_steps"] = 3
    config["asbs_corrector_steps"] = 2

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    assert sampler.asbs_options["adjoint_steps"] == 3
    assert sampler.asbs_options["corrector_steps"] == 2


def test_repo_endpoint_generation_supports_resampling_chunks():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_resample_size"] = 7
    config["asbs_resample_batch_size"] = 3

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    endpoints = sampler._generate_training_endpoints(
        x_initial,
        jax.random.PRNGKey(0),
        resample_size=7,
        resample_batch_size=3,
    )

    assert endpoints.shape == (7, 2)
    assert bool(jnp.all(jnp.isfinite(endpoints)))


def test_repo_endpoint_generation_is_invariant_to_chunk_size_for_same_seed():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
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

    full = sampler._generate_training_endpoints(
        x_initial,
        key,
        resample_size=4,
        resample_batch_size=4,
    )
    chunked = sampler._generate_training_endpoints(
        x_initial,
        key,
        resample_size=4,
        resample_batch_size=2,
    )

    assert full.shape == chunked.shape
    assert bool(jnp.all(jnp.isfinite(full)))
    assert bool(jnp.all(jnp.isfinite(chunked)))


def test_official_repository_training_iterations_are_resolved():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_train_iterations_per_epoch"] = 37

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    assert sampler.asbs_options["train_iterations"] == 37


def test_official_repository_resampling_options_are_resolved():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
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
        h=1.0 / 5,
        T=5,
    )

    assert sampler.asbs_options["resample_size"] == 200
    assert sampler.asbs_options["resample_batch_size"] == 50
    assert sampler.asbs_options["train_batch_size"] == 25


def test_official_repository_resampling_options_must_be_positive():
    for name in (
        "asbs_resample_size",
        "asbs_resample_batch_size",
        "asbs_train_batch_size",
    ):
        config = build_config()
        config["asbs_implementation"] = "official_repository"
        config[name] = 0

        target = IsotropicGaussianTarget(
            mean=(0.0, 0.0),
            variance=1.0,
        )

        try:
            SchrodingerBridgeSampler(
                config=config,
                densitycl=target,
                h=1.0 / 5,
                T=5,
            )
        except ValueError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f"Expected {name}=0 to fail")


def test_official_repository_replay_capacity_is_enforced():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 4

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.zeros((3, 2))

    sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
    )
    result = sampler._train_official_repository_adjoint_stage(
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


def test_official_repository_accepts_geometric_reference_process():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
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

    assert sampler.asbs_options["implementation"] == "official_repository"
    assert sampler.asbs_options["diffusion_schedule"] == "geometric"

    t = jnp.array([0.0, sampler.terminal_time / 2.0, sampler.terminal_time])

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
    assert bool(jnp.all(times < sampler.terminal_time))


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
            1.0,
            include_corrector=False,
        ),
        -jnp.asarray(sampler.eval_sigma(1.0)) * expected,
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
    expected_target = -jnp.asarray(sampler.eval_sigma(1.0)) * expected_adjoint

    np.testing.assert_allclose(
        sampler.eval_terminal_adjoint(x),
        expected_adjoint,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        sampler.eval_adjoint_target(x, 1.0),
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

    history = sampler._train_paper_stage(
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


def test_control_parameterization_is_fixed_by_asbs_implementation():
    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    paper_config = build_config()
    paper_config["asbs_implementation"] = "paper"

    paper_sampler = SchrodingerBridgeSampler(
        config=paper_config,
        densitycl=target,
        h=0.1,
        T=10,
    )

    repo_config = build_config()
    repo_config["asbs_implementation"] = "official_repository"

    repo_sampler = SchrodingerBridgeSampler(
        config=repo_config,
        densitycl=target,
        h=0.1,
        T=10,
    )

    assert paper_sampler.asbs_options["control_parameterization"] == "diffusion"
    assert repo_sampler.asbs_options["control_parameterization"] == "diffusion_squared"


def test_control_parameterization_cannot_override_implementation():
    config = build_config()
    config["asbs_implementation"] = "paper"
    config["asbs_control_parameterization"] = "diffusion_squared"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    try:
        SchrodingerBridgeSampler(
            config=config,
            densitycl=target,
            h=0.1,
            T=10,
        )
    except ValueError as exc:
        assert "asbs_control_parameterization is not configurable" in str(exc)
        assert "fixed by asbs_implementation" in str(exc)
    else:
        raise AssertionError("Expected a control-parameterization override to fail")


def test_adjoint_targets_follow_implementation_profile():
    sigma = 0.4
    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    def make_sampler(mode):
        config = build_config()
        config["asbs_implementation"] = mode

        return SchrodingerBridgeSampler(
            config=config,
            densitycl=target,
            h=0.1,
            T=10,
            sigma_schedule=lambda t: sigma + 0.0 * t,
            integrated_variance=lambda t: sigma**2 * t,
        )

    paper_sampler = make_sampler("paper")
    repo_sampler = make_sampler("official_repository")

    x_terminal = jnp.array(
        [
            [0.5, -1.0],
            [1.5, 0.25],
        ]
    )
    times = jnp.array([0.2, 0.8])

    terminal_adjoint = x_terminal

    np.testing.assert_allclose(
        paper_sampler.eval_adjoint_target(
            x_terminal,
            times,
            include_corrector=False,
        ),
        -sigma * terminal_adjoint,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        repo_sampler.eval_adjoint_target(
            x_terminal,
            times,
            include_corrector=False,
        ),
        -terminal_adjoint,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_implementation_profiles_represent_consistent_physical_drift():
    sigma = 0.4
    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    def make_sampler(mode):
        config = build_config()
        config["asbs_implementation"] = mode

        return SchrodingerBridgeSampler(
            config=config,
            densitycl=target,
            h=0.1,
            T=10,
            sigma_schedule=lambda t: sigma + 0.0 * t,
            integrated_variance=lambda t: sigma**2 * t,
        )

    paper_sampler = make_sampler("paper")
    repo_sampler = make_sampler("official_repository")

    paper_sampler.controller = lambda x, t: sigma * jnp.ones_like(x)
    repo_sampler.controller = lambda x, t: jnp.ones_like(x)

    x = jnp.zeros((2, 2))
    times = jnp.array([0.2, 0.8])
    expected_drift = sigma**2 * jnp.ones_like(x)

    np.testing.assert_allclose(
        paper_sampler.eval_control_drift(x, times),
        expected_drift,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        repo_sampler.eval_control_drift(x, times),
        expected_drift,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_repo_adjoint_stage_refreshes_replay_each_epoch():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 20
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    result = sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=3,
        batch_size=2,
        fresh_samples=2,
    )

    assert result["adjoint_loss"].shape == (3,)
    assert result["buffer_size"] == 6


def test_repo_corrector_stage_refreshes_replay_each_epoch():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 20
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    result = sampler._train_official_repository_corrector_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=3,
        batch_size=2,
        fresh_samples=2,
    )

    assert result["corrector_loss"].shape == (3,)
    assert result["buffer_size"] == 6


def test_repo_source_sampler_is_called_once_per_epoch():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 20
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.zeros((2, 2))
    calls = []

    def source_sampler(key, nsamples):
        del key
        calls.append(nsamples)
        value = float(len(calls))
        return jnp.full((nsamples, 2), value)

    sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=3,
        batch_size=2,
        fresh_samples=2,
        source_sampler=source_sampler,
    )

    assert calls == [2, 2, 2]

    replay = sampler._adjoint_replay.build_dataset()
    expected = jnp.concatenate(
        [
            jnp.full((2, 2), 1.0),
            jnp.full((2, 2), 2.0),
            jnp.full((2, 2), 3.0),
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        replay["x_initial"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_repo_stage_uses_configured_training_batch_size_by_default():
    observed_batch_sizes = []

    class RecordingSampler(SchrodingerBridgeSampler):
        def _train_epoch(
            self,
            model,
            loss_fn,
            optimizer,
            dataset,
            batch_key,
            builder,
            train_batch_size,
        ):
            del model, loss_fn, optimizer, dataset, builder
            observed_batch_sizes.append(train_batch_size)
            return jnp.asarray([0.0]), batch_key

    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_train_batch_size"] = 3

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = RecordingSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
        ]
    )

    sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
        fresh_samples=2,
    )

    assert observed_batch_sizes == [3]


def test_repo_stage_does_not_replace_or_mutate_resolved_options():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_train_batch_size"] = 3
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    options = sampler.asbs_options
    expected = dict(options)

    sampler._train_official_repository_adjoint_stage(
        jnp.zeros((2, 2)),
        jax.random.PRNGKey(0),
        inner_steps=1,
        batch_size=2,
        fresh_samples=2,
    )

    assert sampler.asbs_options is options
    assert sampler.asbs_options == expected


def test_repo_outer_loop_forwards_source_sampler_to_corrector_epochs():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_init_stage"] = "corrector"
    config["asbs_replay_capacity"] = 20
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    calls = []

    def source_sampler(key, nsamples):
        del key
        calls.append(nsamples)
        return jnp.full((nsamples, 2), float(len(calls)))

    history = sampler._train_official_repository_stages(
        jnp.zeros((2, 2)),
        jax.random.PRNGKey(0),
        outer_stages=1,
        corrector_steps=2,
        batch_size=2,
        fresh_samples=2,
        source_sampler=source_sampler,
    )

    assert history["stage"] == ["corrector"]
    assert calls == [2, 2]
    assert history["corrector_buffer_size"] == [4]

    replay = sampler._corrector_replay.build_dataset()
    expected = jnp.concatenate(
        [
            jnp.full((2, 2), 1.0),
            jnp.full((2, 2), 2.0),
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        replay["x_initial"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_paper_mode_rejects_repository_target_clipping():
    config = build_config()
    config["asbs_implementation"] = "paper"
    config["asbs_target_clip"] = 2.0

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    try:
        SchrodingerBridgeSampler(
            config=config,
            densitycl=target,
            h=1.0 / 5,
            T=5,
        )
    except ValueError as exc:
        assert "asbs_target_clip" in str(exc)
        assert "official_repository" in str(exc)
    else:
        raise AssertionError("Expected paper mode to reject repository target clipping")


def test_repo_clips_energy_gradient_before_adding_corrector():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_target_clip"] = 2.0

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    sampler.corrector = lambda x, t: 0.25 * jnp.ones_like(x)

    x = jnp.array(
        [
            [3.0, 4.0],
            [0.3, 0.4],
        ]
    )

    norms = jnp.linalg.norm(x, axis=-1, keepdims=True)
    coefficients = jnp.minimum(
        2.0 / (norms + 1.0e-6),
        1.0,
    )
    expected = coefficients * x + 0.25

    np.testing.assert_allclose(
        sampler.eval_terminal_adjoint(x),
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    # The first energy gradient is clipped, while the small second
    # gradient is unchanged.
    np.testing.assert_allclose(
        expected[1],
        x[1] + 0.25,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_repo_adjoint_loss_uses_elementwise_mean_squared_error():
    config = build_config()
    config["dim"] = 1
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0,),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    class ZeroController(nnx.Module):
        def __call__(self, x, t):
            del t
            return jnp.zeros_like(x)

    x = jnp.array([[0.2], [1.0]])
    t = jnp.array([[0.3], [0.7]])
    labels = jnp.array([[2.0], [-1.0]])

    actual = sampler.compute_adjoint_loss(
        ZeroController(),
        jnp.concatenate([x, t], axis=-1),
        labels,
    )
    expected = jnp.mean(labels**2)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_repo_corrector_loss_uses_elementwise_mean_squared_error():
    config = build_config()
    config["dim"] = 1
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0,),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    class ZeroCorrector(nnx.Module):
        def __call__(self, x, t):
            del t
            return jnp.zeros_like(x)

    x = jnp.array([[0.2], [1.0]])
    labels = jnp.array([[2.0], [-1.0]])

    actual = sampler.compute_corrector_loss(
        ZeroCorrector(),
        x,
        labels,
    )
    expected = jnp.mean(labels**2)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_adjoint_batch_can_explicitly_exclude_corrector():
    sampler = build_sampler(sigma=0.5)
    sampler.corrector = lambda x: 10.0 * jnp.ones_like(x)

    x_initial = jnp.zeros((2, 2))
    x_terminal = jnp.array(
        [
            [0.5, -1.0],
            [1.5, 0.25],
        ]
    )

    batch = sampler.build_adjoint_batch(
        jax.random.PRNGKey(0),
        x_initial,
        x_terminal,
        include_corrector=False,
    )

    times = batch["input"][:, -1]
    expected = -jnp.asarray(sampler.eval_sigma(times))[:, None] * sampler.eval_energy_gradient(
        x_terminal
    )

    np.testing.assert_allclose(
        batch["label"],
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_repo_initial_adjoint_block_excludes_corrector_every_epoch():
    include_corrector_values = []

    class RecordingSampler(SchrodingerBridgeSampler):
        def eval_terminal_adjoint(
            self,
            x_terminal,
            include_corrector=True,
        ):
            include_corrector_values.append(include_corrector)
            return super().eval_terminal_adjoint(
                x_terminal,
                include_corrector=include_corrector,
            )

    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = RecordingSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    sampler._train_official_repository_adjoint_stage(
        jnp.zeros((2, 2)),
        jax.random.PRNGKey(0),
        inner_steps=2,
        batch_size=2,
        fresh_samples=2,
        is_initial_stage=True,
    )

    assert include_corrector_values == [False, False]


def test_repo_stage_schedule_continues_across_train_calls():
    events = []

    class RecordingSampler(SchrodingerBridgeSampler):
        def _train_official_repository_adjoint_stage(
            self,
            *args,
            is_initial_stage=False,
            **kwargs,
        ):
            del args, kwargs
            events.append(("adjoint", is_initial_stage))
            return {
                "adjoint_loss": jnp.asarray([0.0]),
                "buffer_size": len(self._adjoint_replay),
            }

        def _train_official_repository_corrector_stage(
            self,
            *args,
            use_reference_process=False,
            **kwargs,
        ):
            del args, kwargs
            events.append(("corrector", use_reference_process))
            return {
                "corrector_loss": jnp.asarray([0.0]),
                "buffer_size": len(self._corrector_replay),
            }

    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_init_stage"] = "adjoint"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = RecordingSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.zeros((2, 2))

    sampler._train_official_repository_stages(
        x_initial,
        jax.random.PRNGKey(0),
        outer_stages=1,
        adjoint_steps=1,
        corrector_steps=1,
    )
    sampler._train_official_repository_stages(
        x_initial,
        jax.random.PRNGKey(1),
        outer_stages=1,
        adjoint_steps=1,
        corrector_steps=1,
    )

    assert events == [
        ("adjoint", True),
        ("corrector", False),
    ]


def test_paper_stage_tracks_initial_adjoint_semantics():
    include_corrector_values = []

    class RecordingSampler(SchrodingerBridgeSampler):
        def build_adjoint_batch(
            self,
            subkey,
            x_initial,
            x_terminal,
            include_corrector=True,
        ):
            include_corrector_values.append(include_corrector)
            return super().build_adjoint_batch(
                subkey,
                x_initial,
                x_terminal,
                include_corrector=include_corrector,
            )

    config = build_config()
    config["asbs_implementation"] = "paper"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = RecordingSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=5,
    )
    x_initial = jnp.zeros((2, 2))

    sampler._train_paper_stage(
        x_initial,
        jax.random.PRNGKey(0),
        adjoint_steps=2,
        corrector_steps=1,
    )
    sampler._train_paper_stage(
        x_initial,
        jax.random.PRNGKey(1),
        adjoint_steps=1,
        corrector_steps=1,
    )

    assert include_corrector_values == [
        False,
        False,
        True,
    ]


def test_repo_replay_preserves_frozen_terminal_adjoints():
    adjoint_value = [1.0]

    class FrozenAdjointSampler(SchrodingerBridgeSampler):
        def _generate_training_endpoints(
            self,
            x_initial,
            subkey,
            resample_size,
            resample_batch_size,
            reference=False,
        ):
            del subkey, resample_batch_size, reference
            return jnp.asarray(x_initial)[:resample_size]

        def eval_terminal_adjoint(
            self,
            x_terminal,
            include_corrector=True,
        ):
            del include_corrector
            return jnp.full_like(x_terminal, adjoint_value[0])

    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_replay_capacity"] = 20
    config["asbs_train_iterations_per_epoch"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = FrozenAdjointSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x_initial = jnp.zeros((2, 2))

    sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(0),
        inner_steps=1,
        batch_size=2,
        fresh_samples=2,
        is_initial_stage=True,
    )

    adjoint_value[0] = 2.0

    sampler._train_official_repository_adjoint_stage(
        x_initial,
        jax.random.PRNGKey(1),
        inner_steps=1,
        batch_size=2,
        fresh_samples=2,
        is_initial_stage=False,
    )

    replay = sampler._adjoint_replay.build_dataset()

    expected = jnp.concatenate(
        [
            jnp.ones((2, 2)),
            2.0 * jnp.ones((2, 2)),
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        replay["terminal_adjoint"],
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_repo_minibatch_schedule_traverses_shuffle_before_repeating():
    sampler = build_sampler()

    schedule, _ = sampler._build_minibatch_schedule(
        n_data=5,
        batch_key=jax.random.PRNGKey(0),
        train_batch_size=2,
        train_iterations=4,
    )

    assert [indices.shape[0] for indices, _ in schedule] == [
        2,
        2,
        1,
        2,
    ]

    first_cycle = jnp.concatenate(
        [indices for indices, _ in schedule[:3]],
        axis=0,
    )

    np.testing.assert_array_equal(
        jnp.sort(first_cycle),
        jnp.arange(5),
    )

    # The first batch of the next shuffle also contains no replacement.
    assert jnp.unique(schedule[3][0]).shape[0] == 2


def test_repo_minibatch_schedule_keeps_oversized_batch_partial():
    sampler = build_sampler()

    schedule, _ = sampler._build_minibatch_schedule(
        n_data=3,
        batch_key=jax.random.PRNGKey(0),
        train_batch_size=8,
        train_iterations=2,
    )

    assert [indices.shape[0] for indices, _ in schedule] == [3, 3]

    for indices, _ in schedule:
        np.testing.assert_array_equal(
            jnp.sort(indices),
            jnp.arange(3),
        )


def test_repo_minibatch_schedule_is_key_deterministic():
    sampler = build_sampler()
    key = jax.random.PRNGKey(17)

    first, first_key = sampler._build_minibatch_schedule(
        n_data=7,
        batch_key=key,
        train_batch_size=3,
        train_iterations=5,
    )
    second, second_key = sampler._build_minibatch_schedule(
        n_data=7,
        batch_key=key,
        train_batch_size=3,
        train_iterations=5,
    )

    assert len(first) == len(second)

    for (first_indices, first_builder_key), (
        second_indices,
        second_builder_key,
    ) in zip(first, second):
        np.testing.assert_array_equal(
            first_indices,
            second_indices,
        )
        np.testing.assert_array_equal(
            first_builder_key,
            second_builder_key,
        )

    np.testing.assert_array_equal(first_key, second_key)


def test_official_repository_uses_fourier_mlp_architecture():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    assert sampler.controller.channels == 64
    assert sampler.corrector.channels == 64
    assert sampler.controller.num_layers == 4
    assert sampler.corrector.num_layers == 4

    expected_frequencies = jnp.linspace(
        0.1,
        100.0,
        64,
    )[None, :]

    np.testing.assert_allclose(
        sampler.controller.time_embed.timestep_coeff,
        expected_frequencies,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        sampler.corrector.time_embed.timestep_coeff,
        expected_frequencies,
        rtol=0.0,
        atol=0.0,
    )


def test_official_repository_accepts_canonical_model_options():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_model_channels"] = 12
    config["asbs_model_num_layers"] = 3

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    assert sampler.controller.channels == 12
    assert sampler.corrector.channels == 12
    assert sampler.controller.num_layers == 3
    assert sampler.corrector.num_layers == 3

    expected_frequencies = jnp.linspace(
        0.1,
        100.0,
        12,
    )[None, :]

    np.testing.assert_allclose(
        sampler.controller.time_embed.timestep_coeff,
        expected_frequencies,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        sampler.corrector.time_embed.timestep_coeff,
        expected_frequencies,
        rtol=0.0,
        atol=0.0,
    )


def test_official_repository_fourier_models_are_zero_initialized():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    x = jnp.array(
        [
            [0.2, -0.1],
            [1.0, 0.5],
        ]
    )
    t = jnp.array([0.25, 0.75])

    np.testing.assert_allclose(
        sampler.eval_control(x, t),
        jnp.zeros_like(x),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        sampler.eval_corrector(x),
        jnp.zeros_like(x),
        rtol=0.0,
        atol=0.0,
    )


def test_repo_corrector_is_evaluated_at_normalized_terminal_time():
    observed_times = []

    class RecordingCorrector:
        def __call__(self, x, t):
            observed_times.append(t)
            return jnp.zeros_like(x)

    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )
    sampler.corrector = RecordingCorrector()

    x = jnp.zeros((3, 2))
    result = sampler.eval_corrector(x)

    np.testing.assert_allclose(
        result,
        jnp.zeros_like(x),
        rtol=0.0,
        atol=0.0,
    )

    assert len(observed_times) == 1
    np.testing.assert_allclose(
        observed_times[0],
        jnp.ones((3, 1)),
        rtol=0.0,
        atol=0.0,
    )


def test_paper_implementation_retains_configured_rmc_architecture():
    config = build_config()
    config["asbs_implementation"] = "paper"

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

    assert type(sampler.controller).__name__ == "NN_with_time"
    assert type(sampler.corrector).__name__ == "_StaticCorrector"


def test_official_repository_uses_normalized_rmc_time_grid():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sigma = 0.5
    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.2,
        T=5,
        sigma_schedule=lambda t: sigma + 0.0 * t,
        integrated_variance=lambda t: sigma**2 * t,
    )

    assert sampler.terminal_time == 1.0

    np.testing.assert_allclose(
        sampler._integration_time_grid(),
        jnp.linspace(0.0, 1.0, 6),
        rtol=0.0,
        atol=0.0,
    )


def test_official_repository_requires_normalized_terminal_time():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

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
        assert "official_repository" in str(exc)
        assert "h * T == 1" in str(exc)
    else:
        raise AssertionError("Expected a non-normalized official time grid to fail")


def test_official_repository_uses_h_as_integration_step():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sigma = 0.5
    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.2,
        T=5,
        sigma_schedule=lambda t: sigma + 0.0 * t,
        integrated_variance=lambda t: sigma**2 * t,
    )

    grid = sampler._integration_time_grid()

    assert len(grid) == 6
    np.testing.assert_allclose(
        jnp.diff(grid),
        jnp.full((5,), 0.2),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_paper_retains_physical_horizon_and_interval_count():
    config = build_config()
    config["asbs_implementation"] = "paper"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.3,
        T=5,
    )

    assert sampler.terminal_time == 1.5

    np.testing.assert_allclose(
        sampler._integration_time_grid(),
        jnp.linspace(0.0, 1.5, 6),
        rtol=0.0,
        atol=0.0,
    )

    paths = sampler.generate_reference_paths(
        2,
        jax.random.PRNGKey(0),
    )
    assert len(paths) == 6


def test_repo_adjoint_batches_use_normalized_interior_times():
    config = build_config()
    config["asbs_implementation"] = "official_repository"

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=1.0 / 5,
        T=5,
    )

    batch = sampler.build_adjoint_batch(
        jax.random.PRNGKey(0),
        jnp.zeros((100, 2)),
        jnp.ones((100, 2)),
    )
    times = batch["input"][:, -1]

    assert bool(jnp.all(times > 0.0))
    assert bool(jnp.all(times < 1.0))
    assert sampler.terminal_time == 1.0


def test_train_stages_dispatches_paper_cycles():
    config = build_config()
    config["asbs_implementation"] = "paper"
    config["asbs_adjoint_steps"] = 1
    config["asbs_corrector_steps"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.1,
        T=2,
    )

    history = sampler.train_stages(
        jnp.zeros((4, 2)),
        jax.random.PRNGKey(0),
        outer_stages=2,
    )

    assert history["stage"] == [
        "adjoint_corrector",
        "adjoint_corrector",
    ]
    assert len(history["adjoint_loss"]) == 2
    assert len(history["corrector_loss"]) == 2
    assert all(loss.shape == (1,) for loss in history["adjoint_loss"])
    assert all(loss.shape == (1,) for loss in history["corrector_loss"])


def test_train_stages_dispatches_official_repository_blocks():
    config = build_config()
    config["asbs_implementation"] = "official_repository"
    config["asbs_init_stage"] = "corrector"
    config["asbs_train_iterations_per_epoch"] = 1
    config["asbs_replay_duplicates"] = 1

    target = IsotropicGaussianTarget(
        mean=(0.0, 0.0),
        variance=1.0,
    )

    sampler = SchrodingerBridgeSampler(
        config=config,
        densitycl=target,
        h=0.5,
        T=2,
    )

    history = sampler.train_stages(
        jnp.zeros((4, 2)),
        jax.random.PRNGKey(0),
        outer_stages=2,
        adjoint_steps=1,
        corrector_steps=1,
        batch_size=4,
        fresh_samples=4,
    )

    assert history["stage"] == ["corrector", "adjoint"]
    assert len(history["adjoint_loss"]) == 2
    assert len(history["corrector_loss"]) == 2
    assert history["corrector_loss"][0].shape == (1,)
    assert history["adjoint_loss"][1].shape == (1,)

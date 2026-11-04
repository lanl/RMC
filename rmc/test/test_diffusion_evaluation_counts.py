"""Logical target evaluation accounting for PIS and DDS."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from rmc.modules.pis import PathIntegralSampler
from rmc.modules.dds import DenoisingDiffusionSampler


@pytest.mark.parametrize(
    "cls,step_field",
    [
        (PathIntegralSampler, "T"),
        (DenoisingDiffusionSampler, "K"),
    ],
)
@pytest.mark.parametrize("nn_type", ["time_embed", "score"])
@pytest.mark.parametrize("stop_gradient", [False, True])
def test_training_counts(cls, step_field, nn_type, stop_gradient):
    counts = {
        "training": {"density": 0, "gradient": 0, "hvp": 0},
    }
    fake = SimpleNamespace(
        config={
            "nn_type": nn_type,
            "stop_score_gradient": stop_gradient,
        },
        evaluation_counts=counts,
        **{step_field: 4},
    )

    # Two successfully completed minibatches of different sizes.
    cls._record_training_batch(fake, 3)
    cls._record_training_batch(fake, 2)

    score_steps = 4 if nn_type == "score" else 0

    assert counts["training"]["density"] == 5
    assert counts["training"]["gradient"] == 5 * (1 + score_steps)
    assert counts["training"]["hvp"] == (
        5 * score_steps if nn_type == "score" and not stop_gradient else 0
    )


class DummyNetwork:
    def __init__(self, score_informed=False):
        self.score_informed = score_informed
        self.score_calls = 0

    def eval(self):
        pass

    def __call__(self, x, t):
        if self.score_informed:
            self.score_calls += x.shape[0]
        return jnp.zeros_like(x)


class DummyTarget:
    def log_target(self, x):
        return -0.5 * jnp.sum(x**2)


class DummyReference:
    def rvs(self, key, shape):
        return jnp.zeros(shape + (2,))


@pytest.mark.parametrize(
    "cls,step_field,score_informed,expected_density",
    [
        (PathIntegralSampler, "T", False, 5),
        (PathIntegralSampler, "T", True, 5),
        (DenoisingDiffusionSampler, "K", False, 0),
        (DenoisingDiffusionSampler, "K", True, 0),
    ],
)
def test_sampling_counts(cls, step_field, score_informed, expected_density):
    network = DummyNetwork(score_informed)
    steps = 3

    fake = SimpleNamespace(
        d=2,
        h=0.01,
        hsqrt=jnp.sqrt(0.01),
        TT=0.03,
        K=steps,
        T=steps,
        sigma=1.0,
        sigmaSQ=1.0,
        noise_variance=jnp.full((steps,), 0.1),
        retention=jnp.full((steps,), jnp.sqrt(0.9)),
        lmbda=jnp.full((steps,), 1.0 - jnp.sqrt(0.9)),
        ref_process=DummyReference(),
        Dcl=DummyTarget(),
        nnmodel=network,
        config={"nn_type": "score" if score_informed else "time_embed"},
        evaluation_counts={
            "training": {"density": 0, "gradient": 0, "hvp": 0},
            "sampling": {"density": 0, "gradient": 0, "hvp": 0},
        },
    )

    if cls is DenoisingDiffusionSampler:
        fake._control_shift = lambda control, variance, lmbda: (jnp.zeros_like(control))

    # Two generation calls, with 2 and 3 particles.
    cls.sample(fake, 2, jax.random.key(1))
    cls.sample(fake, 3, jax.random.key(2))

    expected_gradient = 5 * steps if score_informed else 0

    assert fake.evaluation_counts["sampling"] == {
        "density": expected_density,
        "gradient": expected_gradient,
        "hvp": 0,
    }
    assert network.score_calls == expected_gradient
    assert fake.evaluation_counts["training"] == {
        "density": 0,
        "gradient": 0,
        "hvp": 0,
    }

"""Tests for LFIS target evaluation accounting."""

import importlib
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rmc.modules.lfis import LiouvilleFlow

lfis_module = importlib.import_module("rmc.modules.lfis")


def zero_counts():
    return {
        "training": {"density": 0, "gradient": 0, "hvp": 0},
        "sampling": {"density": 0, "gradient": 0, "hvp": 0},
    }


class CountingTarget:
    def __init__(self):
        self.density_calls = 0
        self.gradient_calls = 0

    def log_target(self, x):
        self.density_calls += x.shape[0]
        return -0.5 * jnp.sum(x**2, axis=-1)

    def der_log_target_proposal(self, x, tempering):
        self.gradient_calls += x.shape[0]
        return -tempering * x


class ZeroFlow:
    def __call__(self, x):
        return jnp.zeros_like(x)

    def nn_divergence(self, x):
        return jnp.zeros(x.shape[0])


def make_lfis_fake(target):
    fake = SimpleNamespace(
        Dcl=target,
        schedule=lambda t: (t, 1.0),
        LFnn=ZeroFlow(),
        config={"root_path": "unused"},
        tlst=[0.5, 1.0],
        distribution0=lambda key, shape: jnp.ones(shape + (2,)),
        evaluation_counts=zero_counts(),
    )

    for name in (
        "evaluate_score",
        "evaluate_dutlogtarget",
        "evaluate_dutlogtarget_mean",
        "compute_logw_update",
    ):
        setattr(fake, name, MethodType(getattr(LiouvilleFlow, name), fake))

    return fake


def test_training_minibatches():
    fake = SimpleNamespace(evaluation_counts=zero_counts())

    LiouvilleFlow._record_training_batch(fake, 3)
    LiouvilleFlow._record_training_batch(fake, 2)

    assert fake.evaluation_counts["training"] == {
        "density": 5,
        "gradient": 5,
        "hvp": 0,
    }


def test_no_duplicate_density_evaluation():
    target = CountingTarget()
    fake = make_lfis_fake(target)

    x = jnp.ones((4, 2))
    logw = jnp.zeros(4)

    LiouvilleFlow.evaluate_dutlogtarget_mean(fake, x, 0.5, logw)

    assert target.density_calls == 4
    assert target.gradient_calls == 0


def test_sampling_counts_match_target_calls(monkeypatch):
    monkeypatch.setattr(lfis_module, "NN_LiouvilleFlow", lambda config: ZeroFlow())
    monkeypatch.setattr(lfis_module, "load_model", lambda model, root, name: model)

    target = CountingTarget()
    fake = make_lfis_fake(target)

    # Ordinary generation: 3 particles through 2 trained stages.
    path, weights, logz = LiouvilleFlow.sample(fake, 3, True, jax.random.key(1))

    assert len(path) == 3
    assert np.asarray(weights).shape == (3,)
    assert np.isfinite(np.asarray(logz))
    assert fake.evaluation_counts["sampling"] == {
        "density": 6,
        "gradient": 6,
        "hvp": 0,
    }

    # Internal replay sampling belongs to the training phase.
    LiouvilleFlow.sample(fake, 2, False, jax.random.key(2), train=True)

    assert fake.evaluation_counts["training"] == {
        "density": 4,
        "gradient": 4,
        "hvp": 0,
    }

    assert target.density_calls == 10
    assert target.gradient_calls == 10

    # Repeated ordinary sampling must accumulate counts.
    LiouvilleFlow.sample(fake, 1, False, jax.random.key(3))

    assert fake.evaluation_counts["sampling"] == {
        "density": 8,
        "gradient": 8,
        "hvp": 0,
    }
    assert target.density_calls == 12
    assert target.gradient_calls == 12

import jax
import jax.numpy as jnp

import pytest
from flax import nnx

from rmc.modules.lfis import LiouvilleFlow
from rmc.utils.schedule import CosineSchedule

from .test_density import build_density_class


class SetupTest:
    def __init__(self):
        layer_widths = [10, 10]  # number of neurons per layer
        d = 2

        prior_mean = 0.0
        prior_std = 1.0

        self.nn_conf = {
            "seed": 10,
            "task": "train",
            "batch_size": 5,
            "method": "withoutweight",
            "dim": d,
            "layer_widths": layer_widths,
            "activation_func": nnx.relu,
            "opt_type": "ADAM",
            "base_lr": 1e-2,
            "max_epochs": 1,
            "mu0_mean": prior_mean * jnp.ones((1, d)),
            "mu0_covariance": jnp.diagflat((prior_std * jnp.ones((d,))) ** 2).reshape((1, d, d)),
            "dt_max": 1e-1,
            "max_samples": 50,
            "nsamples": 50,
            "eval_every": 1,
            "warm_start": True,
            "max_loss": 5e-2,
            "max_subiter": 1,
            "has_aux": True,
        }


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_create_LFIS(testobj):
    Dcl = build_density_class()
    schedule = CosineSchedule()
    try:
        LFobj = LiouvilleFlow(testobj.nn_conf, Dcl, schedule)
    except Exception as e:
        print(e)
        assert 0


def test_initial_sample_LFIS(testobj):
    key = jax.random.PRNGKey(testobj.nn_conf["seed"])

    Dcl = build_density_class()
    schedule = CosineSchedule()
    LFobj = LiouvilleFlow(testobj.nn_conf, Dcl, schedule)

    t_init = 0.0  # Start interval time
    t = testobj.nn_conf["dt_max"]  # Maximum time step

    try:
        x = LFobj.mu0(key, shape=(testobj.nn_conf["nsamples"],))
    except Exception as e:
        print(e)
        assert 0


def test_dutlt_mean_LFIS(testobj):
    key = jax.random.PRNGKey(testobj.nn_conf["seed"])

    Dcl = build_density_class()
    schedule = CosineSchedule()
    LFobj = LiouvilleFlow(testobj.nn_conf, Dcl, schedule)

    t_init = 0.0  # Start interval time
    t = testobj.nn_conf["dt_max"]  # Maximum time step

    nsamples = testobj.nn_conf["nsamples"]
    x = LFobj.mu0(key, shape=(nsamples,))
    logw = jnp.zeros(nsamples)

    try:
        LFobj.evaluate_dutlogtarget_mean(x, t, logw)
    except Exception as e:
        print(e)
        assert 0


def test_run_empty_LFIS(testobj):
    key = jax.random.PRNGKey(testobj.nn_conf["seed"])

    Dcl = build_density_class()
    schedule = CosineSchedule()
    LFobj = LiouvilleFlow(testobj.nn_conf, Dcl, schedule)
    try:
        LFobj.sample(15, False, key)
    except Exception as e:
        print(e)
        assert 0

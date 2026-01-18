import jax
import jax.numpy as jnp

import pytest
from flax import nnx

from rmc.modules.lfis import LiouvilleFlow
from rmc.utils.density import LogDensityPath
from rmc.utils.packed_distributions import PackedMultivariateNormal
from rmc.utils.schedule import CosineSchedule


def build_density_class():
    d = 2
    cov = 0.3 * jnp.eye(d).reshape((1, d, d))

    class norm2DPath(LogDensityPath):
        def __init__(self, cov):
            self.cov = cov
            self.invcov = jnp.linalg.inv(cov)
            d = cov.shape[-1]
            mean = jnp.zeros((1, d))
            cov_base = 0.5 * jnp.eye(d).reshape((1, d, d))
            self.initial = PackedMultivariateNormal(mean, cov_base)

        def log_initial(self, x):
            return self.initial.log_pdf(x).squeeze()

        def log_target(self, x):
            ll = -0.5 * x @ self.invcov @ x.T
            return ll.squeeze()

    return norm2DPath(cov)


class SetupTest:
    def __init__(self):
        layer_widths = [10, 10]  # number of neurons per layer
        d = 2

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
        LFobj.distribution0(key, shape=(testobj.nn_conf["nsamples"],))
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
    x = LFobj.distribution0(key, shape=(nsamples,))
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

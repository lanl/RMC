import jax
import jax.numpy as jnp

import pytest

from rmc.modules.sampler import HMC, SMC
from rmc.utils.density import BaseLogDensity


class SetupTest:
    def __init__(self):
        self.key = jax.random.key(4444)

        self.N = 20  # Number of particles
        self.d = 2  # dimension
        prior_mean = 0.0
        prior_std = 1.0

        self.T = 32
        sched = jnp.linspace(0, 1, self.T + 1)
        tempering_fn = lambda tstep: sched[tstep]

        density = self.build_density_class()

        self.smp_conf = {
            "seed": 0,
            "sample_shape": (self.N, self.d),
            "initial_sampler_fn": jax.random.multivariate_normal,
            "initial_sampler_mean": prior_mean * jnp.ones((1, self.d)),
            "initial_sampler_covariance": jnp.diagflat(
                (prior_std * jnp.ones((self.d,))) ** 2
            ).reshape((1, self.d, self.d)),
            "maxiter": 1,
            "numsteps": 10,
            "numleapfrog": 20,
            "log_freq": 2,
            "density_cl": density,
            "ESS_thres": 0.98,
            "step_size": 0.01,
            "tempering_fn": tempering_fn,
        }

    def build_density_class(self):
        self.key, subk = jax.random.split(self.key)
        cov = jax.random.normal(self.key, shape=(2, 2))

        class norm2D(BaseLogDensity):
            def __init__(self, cov):
                self.cov = cov
                self.invcov = jnp.linalg.inv(cov)

            def log_target(self, x):
                ll = -0.5 * x @ self.invcov @ x.T
                return ll.squeeze()

        return norm2D(cov)


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_create_HMC(testobj):
    try:
        HMCobj = HMC(testobj.smp_conf)
    except Exception as e:
        print(e)
        assert 0


def test_run_HMC(testobj):
    testobj.smp_conf["sample_shape"] = (1, testobj.d)
    HMCobj = HMC(testobj.smp_conf)
    try:
        HMCobj.sample()
    except Exception as e:
        print(e)
        assert 0


def test_create_SMC(testobj):
    try:
        SMCobj = SMC(testobj.N, testobj.T, testobj.smp_conf)
    except Exception as e:
        print(e)
        assert 0


def test_run_SMC(testobj):
    SMCobj = SMC(testobj.N, testobj.T, testobj.smp_conf)
    try:
        SMCobj.sample()
    except Exception as e:
        print(e)
        assert 0

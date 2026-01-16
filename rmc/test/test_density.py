import jax
import jax.numpy as jnp

import pytest

from rmc.utils.density import (
    BaseLogDensity,
    LinearRegressionDensity,
    LogDensityPath,
    LogDensityPosterior,
)


def test_base_log_density_exception():
    with pytest.raises(TypeError):
        BaseLogDensity()


def test_log_density_path_exception():
    with pytest.raises(TypeError):
        LogDensityPath()


def test_log_density_posterior_exception():
    with pytest.raises(TypeError):
        LogDensityPosterior()


class SetupTest:
    def __init__(self):
        N = 10  # number of (x, y) pairs
        self.D = 2  # problem dimension
        self.noise_std = 0.1  # standard deviation of noise

        # Random generation
        self.key = jax.random.key(0)
        self.key, x_key, y_key = jax.random.split(self.key, 3)
        self.X = jnp.concatenate(
            (jnp.ones((N, 1)), jax.random.normal(x_key, shape=(N, self.D - 1))), 1
        )
        self.true_w = jnp.linspace(0.3, 0.4, self.D)
        self.Y = self.X @ self.true_w + self.noise_std * jax.random.normal(y_key, shape=(N,))

        # Prior distribution
        prior_mean = 0.0
        prior_std = 10.0
        self.prior_mean_vec = prior_mean * jnp.ones((1, self.D))
        self.prior_std_vec = prior_std * jnp.ones((1, self.D))

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


def test_log_density(testobj):
    try:
        density = testobj.build_density_class()
    except Exception as e:
        print(e)
        assert 0


def test_log_linear_regression_density(testobj):
    try:
        dobj = LinearRegressionDensity(
            testobj.D,
            testobj.X,
            testobj.Y,
            testobj.noise_std,
            testobj.prior_mean_vec,
            testobj.prior_std_vec,
        )
    except Exception as e:
        print(e)
        assert 0

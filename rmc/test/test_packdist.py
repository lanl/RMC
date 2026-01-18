import jax
import jax.numpy as jnp
from jax.scipy.stats import uniform

import numpy as np
import pytest
from scipy.stats import multivariate_normal as scipymvnorm
from scipy.stats import norm as scipynorm

from rmc.utils.packed_distributions import (
    BasePackedDistribution,
    PackedMultivariateNormal,
    PackedNormal,
    PackedUniform,
)


class SetupTest:
    def __init__(self):
        class Foo(BasePackedDistribution):
            def __init__(self, scale):
                self.scale = scale

            def log_pdf(self, x):
                return uniform.logpdf(x, scale=self.scale)

            def rvs(self, key, shape):
                return jax.random.uniform(key, shape, maxval=self.scale)

        class Foo2(BasePackedDistribution):
            def __init__(self, scale):
                self.scale = scale

            def rvs(self, key, shape):
                return jax.random.uniform(key, shape, maxval=self.scale)

        self.Foo = Foo
        self.Foo2 = Foo2


def test_base_packed_dist_exception():
    with pytest.raises(TypeError):
        BasePackedDistribution()


@pytest.fixture(scope="module")
def testobj():
    yield SetupTest()


def test_create_Foo(testobj):
    scale = 0.5
    try:
        testobj.Foo(scale)
    except Exception as e:
        print(e)
        assert 0


def test_base_packed_dist_log_pdf_exception(testobj):
    scale = 0.5

    with pytest.raises(TypeError):
        testobj.Foo2(scale)


def test_create_uniform():
    try:
        PackedUniform()
    except Exception as e:
        print(e)
        assert 0


def test_create_normal():
    try:
        PackedNormal()
    except Exception as e:
        print(e)
        assert 0


def test_normal_logpdf():
    # Create normal distribution
    mean = 10.3
    stddev = 0.5
    obj = PackedNormal(mean, stddev)
    # Generate uniform samples
    shape = (100,)
    key = jax.random.key(4444)
    x = jax.random.uniform(key, shape, maxval=2 * mean)
    # Compute log pdf from packed distribution
    logpdf = obj.log_pdf(x)
    # Compute log pdf from scipy
    logpdf_scipy = scipynorm.logpdf(x, loc=mean, scale=stddev)

    np.testing.assert_allclose(logpdf, logpdf_scipy, rtol=1e-6)


def test_normal_sampling():
    # Create normal distribution
    mean = 0.3
    stddev = 0.9
    obj = PackedNormal(mean, stddev)
    # Define shape for sampling
    shape = (10_000_000,)
    # Sample from packed distribution
    key = jax.random.key(4444)
    samples = obj.rvs(key, shape)
    mean_samples = jnp.mean(samples)
    stddev_samples = jnp.std(samples)
    # Sample from scipy
    samples_scipy = scipynorm.rvs(loc=mean, scale=stddev, size=shape)
    mean_samples_scipy = jnp.mean(samples_scipy)
    stddev_samples_scipy = jnp.std(samples_scipy)

    np.testing.assert_allclose(mean_samples, mean_samples_scipy, rtol=5e-3)
    np.testing.assert_allclose(stddev_samples, stddev_samples_scipy, rtol=5e-3)


def test_multivariate_normal_logpdf():
    key = jax.random.key(1234)
    key, subkey = jax.random.split(key)
    # Create multivariate normal distribution
    D = 2
    mean = jax.random.normal(subkey, D)
    cov = jnp.diag(jnp.array([0.5, 0.2]))
    obj = PackedMultivariateNormal(mean, cov)
    # Generate uniform samples
    shape = (100, D)
    x = jax.random.uniform(key, shape)
    # Compute log pdf from packed distribution
    logpdf = obj.log_pdf(x)
    # Compute log pdf from scipy
    logpdf_scipy = scipymvnorm.logpdf(x, mean=mean, cov=cov)

    np.testing.assert_allclose(logpdf, logpdf_scipy, rtol=1e-6)

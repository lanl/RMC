import jax
import jax.numpy as jnp

import numpy as np
import pytest

from rmc.utils.kernels import cdist, cdistSQ
from rmc.utils.math_utils import divergence, divergence_2, divergence_key

xarr = jnp.ones((3, 2))
yarr = -1.0 * jnp.ones((2, 2))


def test_cdist():
    result1 = cdist(xarr, yarr)
    result2 = cdist(yarr, xarr)

    assert result1.shape == (3, 2)
    assert result2.shape == (2, 3)

    np.testing.assert_allclose(result1[:2], result2[:, :2])
    ppdist = 2.0 * np.sqrt(2)  # point-to-point distance
    np.testing.assert_allclose(result1[0], ppdist * jnp.ones(2))


def test_cdistSQ():
    result1 = cdistSQ(xarr, yarr)
    result2 = cdistSQ(yarr, xarr)

    assert result1.shape == (3, 2)
    assert result2.shape == (2, 3)

    np.testing.assert_allclose(result1[:2], result2[:, :2])
    ppdistSQ = 8.0  # point-to-point distance squared
    np.testing.assert_allclose(result1[0], ppdistSQ * jnp.ones(2))


class DivTestObj:
    def __init__(
        self,
    ):
        key = jax.random.key(0)
        N, D = (4, 3)
        self.x = jax.random.normal(key, (D,))
        self.xbatch = jax.random.normal(key, (N, D))

        self.func = lambda x: x**2


@pytest.fixture(scope="module")
def testobj():
    yield DivTestObj()


def test_div(testobj):
    x = testobj.x
    f = testobj.func

    jdiv = divergence(f)(x)
    andiv = 2 * jnp.sum(x)

    np.testing.assert_allclose(jdiv, andiv, rtol=1e-4)


def test_div_batch(testobj):
    xbatch = testobj.xbatch
    f = testobj.func

    jdiv = jax.vmap(divergence(f))(xbatch)
    andiv = 2 * jnp.sum(xbatch, axis=-1)

    np.testing.assert_allclose(jdiv, andiv, rtol=1e-4)


def test_div_2(testobj):
    x = testobj.x
    f = testobj.func

    jdiv = divergence_2(f)(x)
    andiv = 2 * jnp.sum(x)

    np.testing.assert_allclose(jdiv, andiv, rtol=1e-4)


@pytest.mark.parametrize("mode", [-1, 0, 1])
def test_div_key(testobj, mode):
    x = testobj.x
    f = testobj.func

    key = jax.random.key(12345)
    gaussian = False  # True

    jdiv = divergence_key(f, mode, gaussian)(x, key)
    andiv = 2 * jnp.sum(x)

    np.testing.assert_allclose(jdiv, andiv, rtol=1e-4)

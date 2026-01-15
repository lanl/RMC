import jax.numpy as jnp

import numpy as np

from rmc.utils.kernels import cdist, cdistSQ

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

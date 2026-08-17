import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from rmc.flax.models import NN_gradient_informed


def _config(**overrides):
    config = {
        "seed": 0,
        "dim": 2,
        "layer_widths": [8, 8],
        "layer_widths_t": [8, 8],
        "activation_func": nnx.silu,
    }
    config.update(overrides)
    return config


def _linear_score(x):
    return 2.0 * x


def _quadratic_score(x):
    return x**2


def test_gradient_informed_scalar_score_weight_shape():
    model = NN_gradient_informed(
        _config(score_weight_mode="scalar"),
        _linear_score,
    )

    t = jnp.ones((4, 1))
    weight = model.nn2(t)

    assert weight.shape == (4, 1)


def test_gradient_informed_vector_score_weight_shape():
    model = NN_gradient_informed(
        _config(score_weight_mode="vector"),
        _linear_score,
    )

    t = jnp.ones((4, 1))
    weight = model.nn2(t)

    assert weight.shape == (4, 2)


def test_gradient_informed_zero_initial_control():
    model = NN_gradient_informed(
        _config(
            score_weight_mode="vector",
            zero_init_output=True,
            zero_init_score_weight=True,
        ),
        _linear_score,
    )

    x = jnp.array(
        [
            [1.0, -2.0],
            [0.5, 3.0],
        ]
    )

    actual = model(x, 0.5)

    np.testing.assert_allclose(actual, jnp.zeros_like(x), atol=1e-7)


def test_gradient_informed_stop_score_gradient():
    model = NN_gradient_informed(
        _config(
            score_weight_mode="vector",
            stop_score_gradient=True,
            zero_init_output=True,
            zero_init_score_weight=False,
        ),
        _quadratic_score,
    )

    x = jnp.array([[1.0, 2.0]])

    grad = jax.grad(lambda z: jnp.sum(model(z, 0.5)))(x)

    np.testing.assert_allclose(grad, jnp.zeros_like(x), atol=1e-7)


def test_gradient_informed_score_weight_mode_validation():
    with pytest.raises(ValueError, match="Unsupported score_weight_mode"):
        NN_gradient_informed(
            _config(score_weight_mode="invalid"),
            _linear_score,
        )

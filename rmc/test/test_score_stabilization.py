import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from rmc.flax.models import NN_gradient_informed


def _config(**kwargs):
    config = {
        "seed": 0,
        "dim": 2,
        "layer_widths": [8, 8],
        "layer_widths_t": [8, 8],
        "activation_func": nnx.silu,
        "score_weight_mode": "vector",
        "stop_score_gradient": True,
    }
    config.update(kwargs)
    return config


def _constant_score(value):
    def score_fn(x):
        return jnp.full_like(x, value)

    return score_fn


def _zero_score(x):
    return jnp.zeros_like(x)


def test_score_clip_matches_explicitly_clipped_score():
    config = _config(score_clip=100.0)

    model_large = NN_gradient_informed(
        config,
        _constant_score(1.0e6),
    )
    model_clipped = NN_gradient_informed(
        config,
        _constant_score(100.0),
    )

    x = jnp.array(
        [
            [0.25, -0.50],
            [1.00, 0.75],
        ]
    )

    out_large = model_large(x, 0.5)
    out_clipped = model_clipped(x, 0.5)

    np.testing.assert_allclose(
        out_large,
        out_clipped,
        rtol=1e-6,
        atol=1e-6,
    )


def test_state_output_clip_matches_clipped_unbounded_state():
    raw_model = NN_gradient_informed(
        _config(),
        _zero_score,
    )
    clipped_model = NN_gradient_informed(
        _config(state_output_clip=1.0e-6),
        _zero_score,
    )

    x = jnp.array(
        [
            [0.25, -0.50],
            [1.00, 0.75],
        ]
    )

    raw_output = raw_model(x, 0.5)
    clipped_output = clipped_model(x, 0.5)

    np.testing.assert_allclose(
        clipped_output,
        jnp.clip(raw_output, -1.0e-6, 1.0e-6),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    "key",
    [
        "score_clip",
        "state_output_clip",
    ],
)
def test_clip_limits_must_be_positive(key):
    with pytest.raises(ValueError, match=key):
        NN_gradient_informed(
            _config(**{key: 0.0}),
            _zero_score,
        )

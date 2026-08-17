import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from rmc.modules.dds import DenoisingDiffusionSampler


class _ToyDensity:
    def log_target(self, x):
        return -0.5 * jnp.sum(x**2)


def _config():
    return {
        "seed": 0,
        "dim": 2,
        "layer_widths": [8, 8],
        "activation_func": nnx.silu,
        "nn_type": "time_concat",
    }


def _schedule(K):
    return jnp.linspace(0.01, 0.2, K)


def _model(parameterization):
    return DenoisingDiffusionSampler(
        _config(),
        _ToyDensity(),
        sigma=0.7,
        K=4,
        beta_schedule=_schedule,
        control_parameterization=parameterization,
    )


def test_dds_control_parameterization_validation():
    with pytest.raises(
        ValueError,
        match="Unsupported DDS control parameterization",
    ):
        _model("invalid")


def test_dds_control_parameterizations_have_same_mean_shift():
    model_f = _model("f")
    model_u = _model("u")

    noise_variance = jnp.asarray(0.2)
    retention = jnp.sqrt(1.0 - noise_variance)
    lmbda = 1.0 - retention

    f_control = jnp.array(
        [
            [0.5, -1.0],
            [2.0, 0.25],
        ]
    )

    u_control = (
        2.0
        * model_f.sigmaSQ
        * lmbda
        / noise_variance
        * f_control
    )

    shift_f = model_f._control_shift(
        f_control,
        noise_variance,
        lmbda,
    )
    shift_u = model_u._control_shift(
        u_control,
        noise_variance,
        lmbda,
    )

    np.testing.assert_allclose(
        shift_f,
        shift_u,
        rtol=1e-6,
        atol=1e-7,
    )


def test_dds_control_parameterizations_have_same_running_cost():
    model_f = _model("f")
    model_u = _model("u")

    noise_variance = jnp.asarray(0.2)
    retention = jnp.sqrt(1.0 - noise_variance)
    lmbda = 1.0 - retention

    f_control = jnp.array(
        [
            [0.5, -1.0],
            [2.0, 0.25],
        ]
    )

    u_control = (
        2.0
        * model_f.sigmaSQ
        * lmbda
        / noise_variance
        * f_control
    )

    cost_f = model_f._running_cost(
        f_control,
        noise_variance,
        lmbda,
    )
    cost_u = model_u._running_cost(
        u_control,
        noise_variance,
        lmbda,
    )

    np.testing.assert_allclose(
        cost_f,
        cost_u,
        rtol=1e-6,
        atol=1e-7,
    )

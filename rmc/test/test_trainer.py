import jax
import jax.numpy as jnp
from flax import nnx

from rmc.flax.trainer import train


class TinyModel(nnx.Module):
    def __init__(self, seed=0):
        self.linear = nnx.Linear(
            1,
            1,
            rngs=nnx.Rngs(seed),
        )

    def __call__(self, x):
        return self.linear(x)


def _config():
    return {
        "opt_type": "ADAM",
        "base_lr": 1.0e-3,
        "batch_size": 2,
        "eval_every": 1,
        "max_epochs": 1,
        "max_loss": -1.0,
        "has_aux": False,
    }


def _dataset():
    x = jnp.array(
        [
            [-1.0],
            [0.0],
            [1.0],
            [2.0],
        ]
    )
    y = 2.0 * x + 1.0

    return {
        "input": x,
        "label": y,
    }


def _adam_step(optimizer):
    return int(optimizer.opt_state[1].inner_state[0].count.get_value())


def test_train_legacy_return_signature():
    model = TinyModel(seed=0)

    result = train(
        _config(),
        model,
        jax.random.PRNGKey(0),
        _dataset(),
    )

    assert len(result) == 2

    trained_model, loss = result

    assert trained_model is model
    assert jnp.isfinite(loss)


def test_train_reuses_optimizer_state():
    model = TinyModel(seed=0)
    config = _config()
    ds = _dataset()

    model, loss, optimizer = train(
        config,
        model,
        jax.random.PRNGKey(0),
        ds,
        return_optimizer=True,
    )

    assert jnp.isfinite(loss)

    optimizer_identity = optimizer
    step_after_first_call = _adam_step(optimizer)

    assert step_after_first_call > 0

    model, loss, optimizer = train(
        config,
        model,
        jax.random.PRNGKey(1),
        ds,
        optimizer=optimizer,
        return_optimizer=True,
    )

    step_after_second_call = _adam_step(optimizer)

    assert jnp.isfinite(loss)
    assert optimizer is optimizer_identity
    assert step_after_second_call == 2 * step_after_first_call

import jax
import jax.numpy as jnp

import numpy as np
import pytest

from rmc.utils.schedule import (
    BaseSchedule,
    CosineSchedule,
    LinearSchedule,
    QuadraticSchedule,
)


def test_base_schedule_exception():
    with pytest.raises(TypeError):
        BaseSchedule()


def test_cosine_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = CosineSchedule()

    tau = jax.vmap(lambda x: schedule.tau(x))(t)
    atau = 0.5 * (1.0 - jnp.cos(jnp.pi * t))

    np.testing.assert_allclose(tau, atau)


def test_der_cosine_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = CosineSchedule()

    dtau = jax.vmap(lambda x: schedule.dtau(x))(t)
    adtau = 0.5 * jnp.pi * jnp.sin(jnp.pi * t)

    np.testing.assert_allclose(dtau, adtau)


def test_call_cosine_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = CosineSchedule()

    tau, dtau = jax.vmap(lambda x: schedule(x))(t)
    atau = 0.5 * (1.0 - jnp.cos(jnp.pi * t))
    adtau = 0.5 * jnp.pi * jnp.sin(jnp.pi * t)

    np.testing.assert_allclose(tau, atau)
    np.testing.assert_allclose(dtau, adtau)


def test_linear_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = LinearSchedule()

    tau = jax.vmap(lambda x: schedule.tau(x))(t)
    atau = t

    np.testing.assert_allclose(tau, atau)


def test_der_linear_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = LinearSchedule()

    dtau = jax.vmap(lambda x: schedule.dtau(x))(t)
    adtau = jnp.ones(t.shape)

    np.testing.assert_allclose(dtau, adtau)


def test_call_linear_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = LinearSchedule()

    tau, dtau = jax.vmap(lambda x: schedule(x))(t)
    atau = t
    adtau = jnp.ones(t.shape)

    np.testing.assert_allclose(tau, atau)
    np.testing.assert_allclose(dtau, adtau)


def test_quadratic_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = QuadraticSchedule()

    tau = jax.vmap(lambda x: schedule.tau(x))(t)
    atau = t * t

    np.testing.assert_allclose(tau, atau)


def test_der_quadratic_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = QuadraticSchedule()

    dtau = jax.vmap(lambda x: schedule.dtau(x))(t)
    adtau = 2 * t

    np.testing.assert_allclose(dtau, adtau)


def test_call_quadratic_schedule():

    key = jax.random.key(0)
    N = 10
    t = jax.random.uniform(key, (N,))
    schedule = QuadraticSchedule()

    tau, dtau = jax.vmap(lambda x: schedule(x))(t)
    atau = t * t
    adtau = 2 * t

    np.testing.assert_allclose(tau, atau)
    np.testing.assert_allclose(dtau, adtau)

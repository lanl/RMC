import numpy as np
import pytest

from rmc.utils.schedule_diffusion import (
    cosine_beta_schedule,
    linear_beta_schedule,
    prepare_dds_noise_variance,
)


def test_dds_noise_variance_uses_schedule_directly():
    raw = linear_beta_schedule(
        timesteps=5,
        beta_start=0.01,
        beta_end=0.05,
    )

    actual = prepare_dds_noise_variance(
        raw,
        convention="noise_variance",
        reverse=True,
    )

    np.testing.assert_allclose(actual, np.asarray(raw)[::-1])


def test_dds_noise_variance_can_preserve_schedule_order():
    raw = linear_beta_schedule(
        timesteps=5,
        beta_start=0.01,
        beta_end=0.05,
    )

    actual = prepare_dds_noise_variance(
        raw,
        convention="noise_variance",
        reverse=False,
    )

    np.testing.assert_allclose(actual, raw)


def test_dds_legacy_complement_reproduces_old_rmc_schedule():
    raw = cosine_beta_schedule(25)

    actual = prepare_dds_noise_variance(
        raw,
        convention="legacy_complement",
        reverse=True,
    )

    expected = (1.0 - np.asarray(raw))[::-1]
    np.testing.assert_allclose(actual, expected)


def test_dds_cosine_schedule_reverse_order():
    raw = cosine_beta_schedule(25)

    actual = prepare_dds_noise_variance(
        raw,
        convention="noise_variance",
        reverse=True,
    )

    expected = np.asarray(raw)[::-1]

    np.testing.assert_allclose(actual, expected)
    assert np.all(np.diff(np.asarray(actual)) < 0.0)


def test_dds_cosine_schedule_k25_endpoints():
    raw = cosine_beta_schedule(25)

    actual = prepare_dds_noise_variance(
        raw,
        convention="noise_variance",
        reverse=True,
    )

    np.testing.assert_allclose(actual[0], 0.99989998, rtol=1e-6)
    np.testing.assert_allclose(actual[-1], 0.00542998, rtol=1e-5)


def test_dds_schedule_convention_validation():
    raw = linear_beta_schedule(5)

    with pytest.raises(ValueError, match="Unsupported DDS schedule convention"):
        prepare_dds_noise_variance(
            raw,
            convention="not-a-convention",
        )

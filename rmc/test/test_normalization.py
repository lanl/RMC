"""Tests for log-normalizer error."""

from rmc.metrics.normalization import log_z_error


def test_log_z_error():
    assert log_z_error(2.5, 2.0) == 0.5
    assert log_z_error(2.0, 2.0) == 0.0

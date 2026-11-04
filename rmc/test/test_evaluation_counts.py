"""Tests for standardized target-evaluation accounting."""

from types import SimpleNamespace

import pytest

from rmc.metrics.evaluation_counts import get_evaluation_counts


def test_classical_sampler_counts():
    sampler = SimpleNamespace(evaluation_counts={"density": 12, "gradient": 30})

    result = get_evaluation_counts(sampler)

    assert result["training"] == {"density": 0, "gradient": 0, "hvp": 0}
    assert result["sampling"] == {"density": 12, "gradient": 30, "hvp": 0}
    assert result["total"] == result["sampling"]


def test_neural_sampler_counts():
    sampler = SimpleNamespace(
        evaluation_counts={
            "training": {"density": 100, "gradient": 200, "hvp": 30},
            "sampling": {"density": 10, "gradient": 20, "hvp": 3},
        }
    )

    result = get_evaluation_counts(sampler)

    assert result["training"] == {"density": 100, "gradient": 200, "hvp": 30}
    assert result["sampling"] == {"density": 10, "gradient": 20, "hvp": 3}
    assert result["total"] == {"density": 110, "gradient": 220, "hvp": 33}


def test_missing_hvp_defaults_to_zero():
    sampler = SimpleNamespace(
        evaluation_counts={
            "training": {"gradient": 5},
            "sampling": {"density": 7},
        }
    )

    result = get_evaluation_counts(sampler)

    assert result["total"] == {"density": 7, "gradient": 5, "hvp": 0}


def test_metric_does_not_mutate_sampler():
    original = {"density": 4, "gradient": 8}
    sampler = SimpleNamespace(evaluation_counts=original)

    result = get_evaluation_counts(sampler)
    result["sampling"]["density"] = 999

    assert sampler.evaluation_counts == original


def test_missing_attribute_raises():
    with pytest.raises(AttributeError):
        get_evaluation_counts(SimpleNamespace())


def test_incomplete_phase_structure_raises():
    sampler = SimpleNamespace(evaluation_counts={"training": {"gradient": 2}})

    with pytest.raises(ValueError):
        get_evaluation_counts(sampler)


@pytest.mark.parametrize("value", [-1, 1.5, True, "10"])
def test_invalid_count_raises(value):
    sampler = SimpleNamespace(evaluation_counts={"gradient": value})

    with pytest.raises((TypeError, ValueError)):
        get_evaluation_counts(sampler)

"""Normalize target-evaluation counters across RMC samplers."""

from collections.abc import Mapping

OPERATIONS = ("density", "gradient", "hvp")


def get_evaluation_counts(sampler):
    """Return training, sampling, and total target-evaluation counts.

    Classical samplers expose a flat evaluation_counts dictionary, which
    is interpreted as sampling cost. Neural samplers expose separate
    training and sampling dictionaries.

    Counts represent logical target evaluations per point, not Python
    calls, wall-clock time, or FLOPs. The sampler owns the counters;
    this function returns an independent snapshot.
    """
    if not hasattr(sampler, "evaluation_counts"):
        raise AttributeError("Sampler does not expose evaluation_counts")

    counts = sampler.evaluation_counts
    if not isinstance(counts, Mapping):
        raise TypeError("evaluation_counts must be a mapping")

    def normalize(phase):
        if not isinstance(phase, Mapping):
            raise TypeError("Evaluation counts for a phase must be a mapping")

        result = {}
        for operation in OPERATIONS:
            value = phase.get(operation, 0)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{operation} evaluation count must be an integer")
            if value < 0:
                raise ValueError(f"{operation} evaluation count must be nonnegative")
            result[operation] = value

        return result

    if "training" in counts or "sampling" in counts:
        if "training" not in counts or "sampling" not in counts:
            raise ValueError("Phased evaluation counts require both training and sampling")
        training = normalize(counts["training"])
        sampling = normalize(counts["sampling"])
    else:
        training = normalize({})
        sampling = normalize(counts)

    total = {operation: training[operation] + sampling[operation] for operation in OPERATIONS}

    return {
        "training": training,
        "sampling": sampling,
        "total": total,
    }

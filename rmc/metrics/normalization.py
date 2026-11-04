"""Evaluation of log-normalizing-constant estimates."""


def log_z_error(estimate, reference):
    """Return the absolute error in an estimated log partition function."""
    return abs(estimate - reference)

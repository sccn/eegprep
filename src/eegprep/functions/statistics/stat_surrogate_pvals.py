"""Surrogate empirical p-value helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import as_numeric_array


def stat_surrogate_pvals(distribution: Any, observed: Any, tail: str = "both") -> np.ndarray:
    """Compute empirical p-values against a surrogate distribution.

    Args:
        distribution: Surrogate statistic array whose last axis stores
            surrogate replications.
        observed: Observed statistic array matching ``distribution.shape[:-1]``.
        tail: ``"right"``/``"upper"``/``"one"``, ``"left"``/``"lower"``, or
            ``"both"``.
    """

    surrogates = as_numeric_array(distribution, "distribution")
    observed_values = as_numeric_array(observed, "observed", require_axis=False)
    if surrogates.ndim < 1:
        raise ValueError("distribution must have a surrogate axis")
    if observed_values.shape != surrogates.shape[:-1]:
        raise ValueError("observed shape must match distribution without its last axis")

    n_samples = surrogates.shape[-1]
    expanded = np.expand_dims(observed_values, axis=-1)
    p_right = np.sum(surrogates >= expanded, axis=-1) / n_samples
    tail_name = tail.lower()
    if tail_name in {"right", "upper", "one"}:
        return p_right

    p_left = 1 - p_right + np.sum(surrogates == expanded, axis=-1) / n_samples
    if tail_name in {"left", "lower"}:
        return p_left
    if tail_name == "both":
        return np.minimum(2 * np.minimum(p_right, p_left), 1.0)
    raise ValueError("tail must be 'right', 'upper', 'left', 'lower', 'one', or 'both'")


__all__ = ["stat_surrogate_pvals"]

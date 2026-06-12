"""Surrogate confidence interval helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import as_numeric_array


def stat_surrogate_ci(distribution: Any, alpha: float = 0.05, tail: str = "both") -> np.ndarray:
    """Compute surrogate confidence intervals along the last axis.

    Args:
        distribution: Surrogate statistic array whose last axis stores
            surrogate replications.
        alpha: Type-I error rate.
        tail: ``"upper"``, ``"lower"``, ``"one"``, or ``"both"``.
    """

    values = as_numeric_array(distribution, "distribution")
    if values.ndim < 1:
        raise ValueError("distribution must have a surrogate axis")
    alpha_value = float(alpha)
    if not 0 <= alpha_value <= 1:
        raise ValueError("alpha must be between 0 and 1")

    sorted_values = np.sort(values, axis=-1)
    n_samples = sorted_values.shape[-1]
    ci_alpha = alpha_value / 2 if tail.lower() == "both" else alpha_value
    low = int(np.floor(ci_alpha * n_samples + 0.5))
    high = n_samples - low
    low_index = min(max(low, 0), n_samples - 1)
    high_index = min(max(high - 1, 0), n_samples - 1)
    mean_values = np.mean(sorted_values, axis=-1)

    tail_name = tail.lower()
    if tail_name == "upper":
        lower = mean_values
        upper = sorted_values[..., high_index]
    elif tail_name == "lower":
        lower = sorted_values[..., low_index]
        upper = mean_values
    elif tail_name in {"both", "one"}:
        lower = sorted_values[..., low_index]
        upper = sorted_values[..., high_index]
    else:
        raise ValueError("tail must be 'upper', 'lower', 'one', or 'both'")

    return np.stack((lower, upper), axis=0)


__all__ = ["stat_surrogate_ci"]

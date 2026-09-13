"""Grouped means and uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np

from .uniquef import uniquef


def means(data: Any, groups: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-group means, standard errors, variances, and group IDs."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("data must be a vector or 2-D observations-by-variables matrix")
    if groups is None:
        group_vector = np.ones(values.shape[0], dtype=float)
        group_ids = np.ones(1, dtype=float)
    else:
        group_vector = np.asarray(groups).reshape(-1)
        if group_vector.size != values.shape[0]:
            raise ValueError("groups must contain one label per observation")
        group_ids, _, _ = uniquef(group_vector, sort=True)

    group_means = np.full((group_ids.size, values.shape[1]), np.nan)
    standard_errors = np.full_like(group_means, np.nan)
    variances = np.full_like(group_means, np.nan)
    for row, group_id in enumerate(group_ids):
        subset = values[group_vector == group_id]
        for column in range(values.shape[1]):
            finite = subset[np.isfinite(subset[:, column]), column]
            if finite.size == 0:
                continue
            group_means[row, column] = np.mean(finite)
            if finite.size > 1:
                variances[row, column] = np.var(finite, ddof=1)
                standard_errors[row, column] = np.sqrt(variances[row, column] / finite.size)
    return group_means, standard_errors, variances, group_ids


__all__ = ["means"]

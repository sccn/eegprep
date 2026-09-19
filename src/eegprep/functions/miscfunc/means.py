"""Grouped means and uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np

from .uniquef import uniquef


def means(data: Any, groups: Any | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-group means, standard errors, variances, and group IDs.

    Statistics use only finite observations. This intentionally fixes EEGLAB's
    ``means`` helper, which divides standard errors by the full group size even
    when nonfinite observations were excluded.
    """
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must be numeric")
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

    mean_dtype = np.result_type(values.dtype, float)
    group_means = np.full((group_ids.size, values.shape[1]), np.nan, dtype=mean_dtype)
    standard_errors = np.full((group_ids.size, values.shape[1]), np.nan)
    variances = np.full_like(standard_errors, np.nan)
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

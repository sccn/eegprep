"""Stable unique numeric values with frequencies."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import real_array


def uniquef(groups: Any, sort: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite unique values, counts, and zero-based first indices."""
    values = real_array(groups, "groups").reshape(-1)
    tolerance = np.finfo(float).eps * 10**4
    unique_values: list[float] = []
    counts: list[int] = []
    first_indices: list[int] = []
    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        matches = np.flatnonzero(np.abs(np.asarray(unique_values) - value) < tolerance)
        if matches.size:
            counts[int(matches[0])] += 1
        else:
            unique_values.append(float(value))
            counts.append(1)
            first_indices.append(index)
    result_values = np.asarray(unique_values)
    result_counts = np.asarray(counts, dtype=int)
    result_indices = np.asarray(first_indices, dtype=int)
    if sort:
        order = np.argsort(result_values, kind="stable")
        return result_values[order], result_counts[order], result_indices[order]
    return result_values, result_counts, result_indices


__all__ = ["uniquef"]

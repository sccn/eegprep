"""Apply row matches and polarity corrections."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import integer_array, real_array


def matperm(
    first: Any, second: Any, first_indices: Any, second_indices: Any, correlations: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder and sign-correct rows of ``first`` into ``second`` row order."""
    left = np.asarray(first)
    right = np.asarray(second)
    rows = integer_array(first_indices, "first_indices").reshape(-1)
    destinations = integer_array(second_indices, "second_indices").reshape(-1)
    values = real_array(correlations, "correlations").reshape(-1)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("input matrices must be 2-D with the same number of columns")
    if not (rows.size == destinations.size == values.size):
        raise ValueError("indices and correlations must have equal lengths")
    if np.unique(rows).size != rows.size or np.unique(destinations).size != destinations.size:
        raise ValueError("row matches must be unique")
    if np.any(rows < 0) or np.any(rows >= left.shape[0]):
        raise IndexError("first_indices are out of range")
    if np.any(destinations < 0) or np.any(destinations >= right.shape[0]):
        raise IndexError("second_indices are out of range")
    order = np.argsort(destinations, kind="stable")
    permutation = rows[order]
    signs = np.where(values[order] < 0, -1, 1)
    return left[permutation] * signs[:, None], permutation


__all__ = ["matperm"]

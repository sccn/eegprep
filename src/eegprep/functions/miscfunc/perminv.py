"""Inverse permutations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import integer_array


def perminv(permutation: Any) -> np.ndarray:
    """Return the inverse of a 1-based permutation vector.

    A one-dimensional vector stays one-dimensional. A row stays a row and a
    column stays a column.
    """
    values = integer_array(permutation, "permutation")
    if values.ndim > 2 or (values.ndim == 2 and values.shape[0] > 1 and values.shape[1] > 1):
        raise ValueError("permutation must be a vector")
    row = values.ndim == 2 and values.shape[0] == 1 and values.size > 1
    order = values.reshape(-1)
    expected = np.arange(1, order.size + 1)
    if np.unique(order).size != order.size or np.any(np.sort(order) != expected):
        raise ValueError("input must contain each integer from 1 to n exactly once")
    inverse = np.empty(order.size, dtype=int)
    inverse[order - 1] = expected
    if values.ndim == 1:
        return inverse
    if row:
        return inverse.reshape(1, -1)
    return inverse.reshape(-1, 1)


__all__ = ["perminv"]

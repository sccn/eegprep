"""Permutation and scale normalization of square matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def eyelike(matrix: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute and scale rows so the output diagonal is one."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    row_norms = np.sum(np.abs(values), axis=1)
    if np.any(row_norms == 0):
        raise ValueError("matrix rows must contain at least one nonzero value")
    normalized = values / row_norms[:, None]
    rows, columns = linear_sum_assignment(-np.abs(normalized))
    permutation = np.zeros_like(values)
    permutation[columns, rows] = 1
    diagonal = np.diag(permutation @ values)
    if np.any(diagonal == 0):
        raise ValueError("matrix cannot be permuted to a nonzero diagonal")
    scale = np.diag(1 / diagonal)
    return scale @ permutation @ values, scale, permutation


__all__ = ["eyelike"]

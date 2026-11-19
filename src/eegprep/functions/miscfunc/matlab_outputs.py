"""MATLAB array shapes for values EEGPrep returns.

MATLAB has no 1-D array. A scalar is 1-by-1, a vector is a row or a column,
and ``[]`` is 0-by-0. These helpers only add that shape. They do not change
values.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_row(value: Any) -> np.ndarray:
    """Return ``value`` as a row, or as 1-by-1 when it is a single number."""
    array = np.asarray(value)
    if array.ndim == 0 or array.size == 1:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def as_column(value: Any) -> np.ndarray:
    """Return ``value`` as a column, or as 1-by-1 when it is a single number."""
    array = np.asarray(value)
    if array.ndim == 0 or (array.ndim == 1 and array.size == 1):
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def empty_matrix(dtype: Any = float) -> np.ndarray:
    """Return MATLAB ``[]``."""
    return np.empty((0, 0), dtype=dtype)

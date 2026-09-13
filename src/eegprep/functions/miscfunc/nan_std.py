"""NaN-aware sample standard deviation."""

from __future__ import annotations

from typing import Any

import numpy as np


def nan_std(data: Any, axis: int | None = None) -> np.ndarray | np.floating[Any]:
    """Return sample standard deviations while ignoring NaNs.

    When ``axis`` is omitted, the first non-singleton dimension is used, as in
    MATLAB. A single non-NaN observation yields ``NaN``.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        return np.float64(np.nan)
    selected_axis = _first_nonsingleton_axis(values) if axis is None else axis
    count = np.sum(~np.isnan(values), axis=selected_axis)
    total = np.nansum(values, axis=selected_axis)
    squared = np.nansum(values**2, axis=selected_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = (squared - total**2 / count) / (count - 1)
    return np.where(count > 1, np.sqrt(np.maximum(variance, 0)), np.nan)


def _first_nonsingleton_axis(values: np.ndarray) -> int:
    for axis, length in enumerate(values.shape):
        if length != 1:
            return axis
    return 0


__all__ = ["nan_std"]

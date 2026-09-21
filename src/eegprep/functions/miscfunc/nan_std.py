"""NaN-aware sample standard deviation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import integer_scalar, real_array


def nan_std(data: Any, axis: int | None = None) -> np.ndarray | np.floating[Any]:
    """Return sample standard deviations while ignoring NaNs.

    When ``axis`` is omitted, the first non-singleton dimension is used, as in
    MATLAB. A single non-NaN observation yields ``NaN``.
    """
    values = real_array(data, "data")
    if values.ndim == 0:
        return np.float64(np.nan)
    selected_axis = _first_nonsingleton_axis(values) if axis is None else integer_scalar(axis, "axis")
    count = np.sum(~np.isnan(values), axis=selected_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.nansum(values, axis=selected_axis) / count
        centered = values - np.expand_dims(mean, axis=selected_axis)
        centered = np.where(np.isnan(values), 0.0, centered)
        variance = np.sum(centered**2, axis=selected_axis) / (count - 1)
    return np.where(count > 1, np.sqrt(variance), np.nan)


def _first_nonsingleton_axis(values: np.ndarray) -> int:
    for axis, length in enumerate(values.shape):
        if length != 1:
            return axis
    return 0


__all__ = ["nan_std"]

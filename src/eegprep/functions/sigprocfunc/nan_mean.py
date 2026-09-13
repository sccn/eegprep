"""NaN-aware arithmetic means."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.nan_std import _first_nonsingleton_axis


def nan_mean(data: Any, axis: int | None = None) -> np.ndarray | np.floating[Any]:
    """Return means while ignoring NaNs along the selected dimension."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        return np.float64(values)
    selected_axis = _first_nonsingleton_axis(values) if axis is None else axis
    count = np.sum(~np.isnan(values), axis=selected_axis)
    total = np.nansum(values, axis=selected_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total / count
    return np.where(count > 0, result, np.nan)


__all__ = ["nan_mean"]

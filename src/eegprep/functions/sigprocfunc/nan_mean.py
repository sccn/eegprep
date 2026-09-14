"""NaN-aware arithmetic means."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc._validation import integer_scalar
from eegprep.functions.miscfunc.nan_std import _first_nonsingleton_axis


def nan_mean(data: Any, axis: int | None = None) -> Any:
    """Return means while ignoring NaNs along the selected dimension."""
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must be numeric")
    if values.ndim == 0:
        return values[()]
    selected_axis = _first_nonsingleton_axis(values) if axis is None else integer_scalar(axis, "axis")
    count = np.sum(~np.isnan(values), axis=selected_axis)
    total = np.nansum(values, axis=selected_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total / count
    return np.where(count > 0, result, np.nan)


__all__ = ["nan_mean"]

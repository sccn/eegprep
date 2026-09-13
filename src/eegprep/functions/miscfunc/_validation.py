"""Validation shared by numerical helper ports."""

from __future__ import annotations

from typing import Any

import numpy as np


def integer_scalar(value: Any, name: str) -> int:
    """Return an exact integer scalar without silently truncating input."""
    array = np.asarray(value)
    if array.ndim != 0 or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must be an integer")
    numeric = float(array)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(numeric)


def integer_array(value: Any, name: str) -> np.ndarray:
    """Return exact integer values without silently truncating input."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain integers")
    numeric = array.astype(float)
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.trunc(numeric)):
        raise ValueError(f"{name} must contain integers")
    return numeric.astype(np.intp)


def real_array(value: Any, name: str) -> np.ndarray:
    """Return real floating-point data without discarding imaginary parts."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    return array.astype(float, copy=False)

"""Numeric data limits."""

from __future__ import annotations

from typing import Any

import numpy as np


def datlim(data: Any) -> np.ndarray:
    """Return the minimum and maximum of a nonempty numeric array."""
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must be a numeric array")
    if values.size == 0:
        raise ValueError("data must not be empty")
    return np.asarray([np.min(values), np.max(values)])


__all__ = ["datlim"]

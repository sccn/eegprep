"""Numeric data limits."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import real_array


def datlim(data: Any) -> np.ndarray:
    """Return the minimum and maximum of a nonempty numeric array."""
    values = real_array(data, "data")
    if values.size == 0:
        raise ValueError("data must not be empty")
    return np.asarray([np.min(values), np.max(values)])


__all__ = ["datlim"]

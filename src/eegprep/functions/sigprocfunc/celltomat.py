"""Numeric cell-like data conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def celltomat(cells: Any) -> np.ndarray:
    """Convert a rectangular nested sequence of numeric scalars to an array."""
    result = np.asarray(cells)
    if result.dtype == object and result.size:
        raise TypeError("cells must form a rectangular numeric array")
    if not np.issubdtype(result.dtype, np.number) and result.size:
        raise TypeError("cells must contain numeric values")
    return result


__all__ = ["celltomat"]

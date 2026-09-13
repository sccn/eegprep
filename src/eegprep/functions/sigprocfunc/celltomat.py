"""Numeric cell-like data conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def celltomat(cells: Any) -> np.ndarray:
    """Convert a rectangular nested sequence of numeric scalars to an array."""
    result = np.asarray(cells)
    if result.dtype == object and result.size:
        values = [_numeric_scalar(cell) for cell in result.flat]
        result = np.asarray(values).reshape(result.shape)
    if not np.issubdtype(result.dtype, np.number) and result.size:
        raise TypeError("cells must contain numeric values")
    return result


def _numeric_scalar(cell: Any) -> Any:
    value = np.asarray(cell)
    if value.size != 1 or not np.issubdtype(value.dtype, np.number):
        raise TypeError("cells must form a rectangular array of numeric scalars")
    return value.reshape(-1)[0]


__all__ = ["celltomat"]

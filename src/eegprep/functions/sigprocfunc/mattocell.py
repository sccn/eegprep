"""Matrix to cell-like data conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def mattocell(matrix: Any) -> np.ndarray:
    """Return a cell array whose entries are 1-by-1 copies of ``matrix``."""
    values = np.asarray(matrix)
    if values.size == 0:
        return np.empty((0, 0), dtype=object)
    cells = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        cells[index] = np.array([[values[index]]])
    return cells


__all__ = ["mattocell"]

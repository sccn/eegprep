"""Matrix to cell-like data conversion."""

from __future__ import annotations

from typing import Any

import numpy as np


def mattocell(matrix: Any) -> list[Any]:
    """Convert a NumPy-compatible matrix to nested Python scalar lists."""
    return np.asarray(matrix).tolist()


__all__ = ["mattocell"]

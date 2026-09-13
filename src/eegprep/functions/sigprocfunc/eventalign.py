"""Score event-array alignment factors."""

from __future__ import annotations

from typing import Any

import numpy as np


def eventalign(factor: Any, a: Any, b: Any, measure: str = "median") -> float:
    """Return the mean or median nearest alignment error.

    A scalar factor scales ``a``. A two-value factor supplies ``(scale,
    offset)``. For matrices, the minimum absolute error is found in each row
    before the row minima are summarized.
    """
    factors = np.asarray(factor, dtype=float).reshape(-1)
    if factors.size not in {1, 2} or not np.isfinite(factors).all():
        raise ValueError("eventalign: factor must contain a finite scale and optional offset")
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.size == 0 or first.shape != second.shape:
        raise ValueError("eventalign: a and b must be non-empty arrays with matching shapes")

    difference = np.abs(factors[0] * first - second + (factors[1] if factors.size == 2 else 0))
    minima = np.asarray(np.min(difference, axis=-1) if difference.ndim > 1 else np.min(difference))
    statistic = str(measure).casefold()
    if statistic == "mean":
        return float(np.mean(minima))
    if statistic == "median":
        return float(np.median(minima))
    raise ValueError("eventalign: measure must be 'mean' or 'median'")


__all__ = ["eventalign"]

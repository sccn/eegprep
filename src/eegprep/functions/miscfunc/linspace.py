"""MATLAB-compatible linearly spaced vectors."""

from __future__ import annotations

import numpy as np


def linspace(start: float, stop: float, count: float = 100) -> np.ndarray:
    """Return ``count`` values from ``start`` through ``stop``.

    Interior points are ``start + i * (stop - start) / (count - 1)``.
    The last value is ``stop``. A count of 1 returns ``stop``.
    """
    n = int(np.floor(count))
    stop_value = np.float64(stop)
    if n < 1:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.asarray([stop_value], dtype=np.float64)
    start_value = np.float64(start)
    index = np.arange(n - 1, dtype=np.float64)
    values = np.empty(n, dtype=np.float64)
    values[:-1] = start_value + index * (stop_value - start_value) / np.float64(n - 1)
    values[-1] = stop_value
    return values

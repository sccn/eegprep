"""Legacy EEGLAB covariance estimator."""

from __future__ import annotations

from typing import Any

import numpy as np


def covary(data: Any) -> np.ndarray | np.floating[Any]:
    """Return EEGLAB's globally centered, unbiased column second moment.

    Unlike :func:`numpy.var`, the historical EEGLAB helper subtracts one
    grand mean from the entire input before computing each column. This
    behavior is retained because existing analyses can depend on it.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        return np.float64(np.nan)
    if values.ndim == 1 or (values.ndim == 2 and values.shape[0] == 1):
        vector = values.reshape(-1)
        if vector.size < 2:
            return np.float64(np.nan)
        centered = vector - np.mean(vector)
        return np.sum(centered * centered) / (vector.size - 1)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("data must be a vector or 2-D matrix with at least two rows")
    centered = values - np.mean(values)
    return np.sum(centered * centered, axis=0) / (values.shape[0] - 1)


__all__ = ["covary"]

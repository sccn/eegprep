"""EEGLAB-compatible excess kurtosis helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def kurt(data: Any) -> np.ndarray:
    """Return column-wise excess kurtosis using EEGLAB's legacy formula."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError("data must be 1-D or 2-D")
    if values.shape[0] == 1:
        values = values.T
    centered = values - np.mean(values, axis=0, keepdims=True)
    m2 = np.mean(centered**2, axis=0)
    m2[m2 == 0] = np.inf
    return np.mean(centered**4, axis=0) / m2**2 - 3.0


__all__ = ["kurt"]

"""Pairwise correlation helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import two_arrays


def corrcoef_cell(a: Any, b: Any | None = None, *, axis: int = -1) -> np.ndarray:
    """Compute pairwise correlations along a case axis."""

    first, second = two_arrays(a, b, "corrcoef_cell", axis=axis)
    if first.shape != second.shape:
        raise ValueError("corrcoef_cell requires arrays with identical shapes")
    if first.shape[-1] < 2:
        raise ValueError("corrcoef_cell requires at least two cases")

    first_centered = first - np.mean(first, axis=-1, keepdims=True)
    second_centered = second - np.mean(second, axis=-1, keepdims=True)
    covariance = np.sum(first_centered * second_centered, axis=-1)
    first_power = np.sum(first_centered * first_centered, axis=-1)
    second_power = np.sum(second_centered * second_centered, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / np.sqrt(first_power * second_power)


__all__ = ["corrcoef_cell"]

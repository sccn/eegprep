"""False discovery rate helper."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FDRResult:
    """False discovery rate threshold and mask."""

    threshold: np.ndarray | float
    mask: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray | float]:
        yield self.threshold
        yield self.mask


def fdr(pvals: Any, q: float | None = None, fdr_type: str = "parametric") -> FDRResult:
    """Compute Benjamini-Hochberg or Benjamini-Yekutieli FDR thresholds.

    Args:
        pvals: Numeric p-value array with values in the closed interval [0, 1].
        q: Desired false discovery rate. If omitted, an EEGLAB-style array of
            corrected thresholds is returned.
        fdr_type: ``"parametric"`` for Benjamini-Hochberg or
            ``"nonparametric"``/``"nonParametric"`` for Benjamini-Yekutieli.

    Returns:
        Threshold and boolean mask with the same shape as ``pvals``.
    """

    values = np.asarray(pvals)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("pvals must be numeric")
    if values.size == 0:
        return FDRResult(np.array([], dtype=float), np.array([], dtype=bool))
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if np.any((finite_values < 0) | (finite_values > 1)):
        raise ValueError("pvals must contain probabilities between 0 and 1")

    if q is None:
        threshold = np.ones(values.shape, dtype=float)
        thresholds = np.exp(np.linspace(np.log(0.1), np.log(0.000001), 1000))
        for current in thresholds:
            current_result = fdr(values, float(current), fdr_type=fdr_type)
            threshold[current_result.mask] = current
        return FDRResult(threshold, finite_mask & (values <= threshold))

    q_value = float(q)
    if not 0 <= q_value <= 1:
        raise ValueError("q must be between 0 and 1")

    fdr_type_name = fdr_type.lower()
    if fdr_type_name not in {"parametric", "nonparametric"}:
        raise ValueError("fdr_type must be 'parametric' or 'nonparametric'")

    if finite_values.size == 0:
        return FDRResult(0.0, np.zeros(values.shape, dtype=bool))

    flat = np.sort(finite_values.reshape(-1))
    count = flat.size
    indices = np.arange(1, count + 1, dtype=float)
    correction = 1.0 if fdr_type_name == "parametric" else float(np.sum(1.0 / indices))
    accepted = flat <= indices / count * q_value / correction
    threshold_value = float(flat[np.flatnonzero(accepted).max()]) if np.any(accepted) else 0.0
    return FDRResult(threshold_value, finite_mask & (values <= threshold_value))


__all__ = ["FDRResult", "fdr"]

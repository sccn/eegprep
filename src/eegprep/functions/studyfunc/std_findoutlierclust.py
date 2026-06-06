"""Find outlier rows in STUDY cluster feature matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def std_findoutlierclust(data: Any, labels: Any | None = None, *, threshold: float = 3.0) -> np.ndarray:
    """Return 1-based row indices farther than ``threshold`` cluster spreads."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D matrix")
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")
    if labels is None:
        labels = np.ones(values.shape[0], dtype=int)
    labels = np.asarray(labels, dtype=int).ravel()
    if labels.size != values.shape[0]:
        raise ValueError("labels length must match data rows")
    outliers = []
    for label in sorted(value for value in set(labels.tolist()) if value > 0):
        rows = np.flatnonzero(labels == label)
        if rows.size < 2:
            continue
        centroid = np.mean(values[rows], axis=0)
        distances = np.linalg.norm(values[rows] - centroid, axis=1)
        spread = float(np.std(distances))
        if spread == 0:
            continue
        outliers.extend((rows[distances > np.mean(distances) + threshold * spread] + 1).tolist())
    return np.asarray(outliers, dtype=int)


__all__ = ["std_findoutlierclust"]

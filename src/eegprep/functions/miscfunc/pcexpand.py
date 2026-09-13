"""Principal-component reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np


def pcexpand(projections: Any, eigenvectors: Any, data_means: Any) -> np.ndarray:
    """Expand component projections back into channel space."""
    projected = np.asarray(projections)
    vectors = np.asarray(eigenvectors)
    means = np.asarray(data_means).reshape(-1)
    if not all(np.issubdtype(array.dtype, np.number) for array in (projected, vectors, means)):
        raise TypeError("PCA inputs must be numeric")
    if projected.ndim != 2 or vectors.ndim != 2 or vectors.shape[0] != vectors.shape[1]:
        raise ValueError("projections must be 2-D and eigenvectors must be square")
    if projected.shape[0] > vectors.shape[1]:
        raise ValueError("eigenvectors must span every projected component")
    if means.size != vectors.shape[0]:
        raise ValueError("data_means must contain one value per output channel")
    return vectors[:, : projected.shape[0]] @ projected + means[:, None]


__all__ = ["pcexpand"]

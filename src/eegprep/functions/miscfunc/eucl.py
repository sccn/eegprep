"""Euclidean distances between point sets."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import real_array
from scipy.spatial.distance import cdist, pdist, squareform


def eucl(coordinates: Any, other: Any | None = None) -> np.ndarray | float:
    """Compute pairwise Euclidean distances between rows of point arrays."""
    first = _points(coordinates)
    if other is None:
        if first.shape[0] < 2:
            raise ValueError("at least two points are required")
        distances = squareform(pdist(first, metric="euclidean"))
        return float(distances[0, 1]) if first.shape[0] == 2 else distances
    second = _points(other)
    if first.shape[1] != second.shape[1]:
        raise ValueError("coordinate sets must have the same dimension")
    distances = cdist(first, second, metric="euclidean")
    if distances.shape == (1, 1):
        return float(distances[0, 0])
    if first.shape[0] == 1:
        return distances[0]
    if second.shape[0] == 1:
        return distances[:, 0]
    return distances


def _points(value: Any) -> np.ndarray:
    points = real_array(value, "coordinates")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        raise ValueError("coordinates must be a point vector or 2-D point array")
    return points


__all__ = ["eucl"]

"""Compute STUDY cluster centroids."""

from __future__ import annotations

from typing import Any

import numpy as np


def std_centroid(data: Any, labels: Any | None = None) -> np.ndarray:
    """Return centroids for all rows or for each positive label."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D matrix")
    if labels is None:
        return np.nanmean(values, axis=0, keepdims=True)
    cluster_labels = np.asarray(labels, dtype=int).ravel()
    if cluster_labels.size != values.shape[0]:
        raise ValueError("labels length must match data rows")
    centroids = []
    for label in sorted(value for value in set(cluster_labels.tolist()) if value > 0):
        centroids.append(np.nanmean(values[cluster_labels == label], axis=0))
    return np.asarray(centroids, dtype=float)


__all__ = ["std_centroid"]

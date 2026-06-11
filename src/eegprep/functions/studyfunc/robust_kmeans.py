"""Deterministic robust k-means for STUDY component clustering."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc.optimal_kmeans import optimal_kmeans
from eegprep.functions.studyfunc.pop_clust import _kmeans_labels, _squared_distances


def robust_kmeans(
    data: Any,
    clus_num: int,
    STD: float = 3.0,
    MAXiter: int = 10,
    method: str = "kmeans",
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster rows and iteratively mark distant rows as outliers."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("data must be a 2-D matrix with at least two rows")
    if method.lower() not in {"kmeans", "kmeanscluster", "optimal_kmeans"}:
        raise ValueError("method must be 'kmeans', 'kmeanscluster', or 'optimal_kmeans'")
    cluster_count = int(clus_num)
    active = np.arange(values.shape[0], dtype=int)
    outliers = np.asarray([], dtype=int)
    labels = np.zeros(values.shape[0], dtype=int)
    centers = np.empty((0, values.shape[1]), dtype=float)
    distances = np.empty((values.shape[0], 0), dtype=float)
    for _iteration in range(max(1, int(MAXiter))):
        if active.size < cluster_count:
            break
        if method.lower() == "optimal_kmeans":
            active_labels, centers, _sumd, _distances = optimal_kmeans(
                values[active], [2, cluster_count], random_state=random_state
            )
        else:
            active_labels, centers = _kmeans_labels(values[active], cluster_count, random_state)
        distances = np.sqrt(_squared_distances(values, centers))
        labels[:] = 0
        labels[active] = active_labels
        new_outliers = _outlier_rows(active, active_labels, distances[active], float(STD))
        if new_outliers.size == 0:
            break
        combined = np.unique(np.concatenate([outliers, new_outliers]))
        if np.array_equal(combined, outliers):
            break
        outliers = combined
        active = np.asarray([index for index in range(values.shape[0]) if index not in set(outliers.tolist())])
    sumd = _sum_distances(labels, distances)
    return labels, centers, sumd, distances, outliers + 1


def _outlier_rows(active: np.ndarray, labels: np.ndarray, distances: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold):
        return np.asarray([], dtype=int)
    if threshold <= 0:
        raise ValueError("STD must be greater than 0")
    reference = []
    selected: list[int] = []
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        if rows.size == 0:
            continue
        current = distances[rows, label - 1]
        reference.append(float(np.mean(current)))
        spread = float(np.std(current))
        if spread == 0:
            continue
        selected.extend(active[rows[current > spread * threshold]].tolist())
    if not reference:
        return np.asarray([], dtype=int)
    guard = float(np.mean(reference)) * threshold
    output = []
    for row in selected:
        position = int(np.where(active == row)[0][0])
        if distances[position, labels[position] - 1] > guard:
            output.append(row)
    return np.asarray(output, dtype=int)


def _sum_distances(labels: np.ndarray, distances: np.ndarray) -> np.ndarray:
    sums = []
    for label in sorted(value for value in set(labels.tolist()) if value > 0):
        rows = np.flatnonzero(labels == label)
        sums.append(float(np.sum(distances[rows, label - 1])))
    return np.asarray(sums, dtype=float)


__all__ = ["robust_kmeans"]

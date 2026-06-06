"""Select a deterministic k-means cluster count for STUDY data."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc.pop_clust import _kmeans_labels, _squared_distances


def optimal_kmeans(
    clustdata: Any, clusnum: int | list[int] | tuple[int, int], *, random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run k-means for a range and choose the best silhouette score."""
    data = np.asarray(clustdata, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("clustdata must be a 2-D matrix with at least two rows")
    values = _cluster_range(clusnum, data.shape[0])
    best = None
    for cluster_count in values:
        labels, centers = _kmeans_labels(data, cluster_count, random_state)
        score = _silhouette_score(data, labels)
        distances = np.sqrt(_squared_distances(data, centers))
        sumd = _sum_distances(labels, distances)
        candidate = (score, labels, centers, sumd, distances)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ValueError("No valid k-means cluster count was available")
    _score, labels, centers, sumd, distances = best
    return labels, centers, sumd, distances


def _cluster_range(clusnum: int | list[int] | tuple[int, int], maximum: int) -> list[int]:
    if isinstance(clusnum, (list, tuple)):
        if len(clusnum) != 2:
            raise ValueError("clusnum must be an integer or [min, max]")
        lower, upper = int(clusnum[0]), int(clusnum[1])
    else:
        lower = upper = int(clusnum)
    lower = max(2, lower)
    upper = min(maximum, upper)
    if lower > upper:
        raise ValueError("clusnum range does not contain a valid cluster count")
    return list(range(lower, upper + 1))


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    distance = np.sqrt(_squared_distances(data, data))
    scores = []
    for row, label in enumerate(labels):
        same = labels == label
        if np.sum(same) <= 1:
            scores.append(0.0)
            continue
        same[row] = False
        a_value = float(np.mean(distance[row, same])) if np.any(same) else 0.0
        b_values = [
            float(np.mean(distance[row, labels == other]))
            for other in sorted(set(labels.tolist()))
            if other != label and np.any(labels == other)
        ]
        if not b_values:
            scores.append(0.0)
            continue
        b_value = min(b_values)
        denom = max(a_value, b_value)
        scores.append(0.0 if denom == 0 else (b_value - a_value) / denom)
    return float(np.mean(scores))


def _sum_distances(labels: np.ndarray, distances: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.sum(distances[np.flatnonzero(labels == label), label - 1]) for label in sorted(set(labels.tolist()))],
        dtype=float,
    )


__all__ = ["optimal_kmeans"]

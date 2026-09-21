"""Deterministic k-means clustering for row-wise observations."""

from __future__ import annotations

from typing import Any

import numpy as np


KMEANS_MAX_ITER = 300
KMEANS_TOLERANCE = 1e-8


def kmeans_st(
    data: Any,
    clusters: int,
    restarts: int = 0,
    *,
    random_state: int = 0,
    max_iter: int = KMEANS_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Partition observations into clusters with minimum within-cluster SSE.

    Observations are rows and features are columns. Cluster labels are zero-based,
    following Python indexing. The first solution and every requested restart use
    independent deterministic initial centers derived from ``random_state``.

    Args:
        data: Two-dimensional observation matrix.
        clusters: Number of non-empty clusters.
        restarts: Number of additional random initializations.
        random_state: Seed used for center initialization.
        max_iter: Maximum Lloyd iterations per initialization.

    Returns:
        Centroids, zero-based membership labels, and total squared error.
    """
    matrix = _validated_observations(data, clusters)
    if int(restarts) != restarts or restarts < 0:
        raise ValueError("restarts must be a non-negative integer")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    labels, centers = _kmeans_labels(
        matrix,
        int(clusters),
        random_state=int(random_state),
        n_init=int(restarts) + 1,
        max_iter=int(max_iter),
    )
    distances = squared_distances(matrix, centers)
    sse = float(np.sum(distances[np.arange(matrix.shape[0]), labels]))
    return centers, labels, sse


def squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances from every row to every center."""
    differences = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(differences * differences, axis=2)


def _kmeans_labels(
    data: np.ndarray,
    clusters: int,
    *,
    random_state: int,
    n_init: int,
    max_iter: int = KMEANS_MAX_ITER,
    initial_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")

    for attempt in range(n_init):
        if attempt == 0 and initial_centers is not None:
            centers = np.asarray(initial_centers, dtype=float).copy()
        else:
            centers = data[rng.choice(data.shape[0], size=clusters, replace=False)].copy()
        labels = np.zeros(data.shape[0], dtype=int)
        for _iteration in range(max_iter):
            labels = np.argmin(squared_distances(data, centers), axis=1)
            new_centers = _recompute_centers(data, labels, centers, clusters)
            if np.allclose(new_centers, centers, rtol=0, atol=KMEANS_TOLERANCE):
                centers = new_centers
                labels = np.argmin(squared_distances(data, centers), axis=1)
                break
            centers = new_centers

        labels = np.argmin(squared_distances(data, centers), axis=1)
        distances = squared_distances(data, centers)
        inertia = float(np.sum(distances[np.arange(data.shape[0]), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:
        raise ValueError("k-means failed to initialize clusters")
    return _canonicalize_clusters(best_labels, best_centers)


def _validated_observations(data: Any, clusters: int) -> np.ndarray:
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("data must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("data must contain only finite values")
    if int(clusters) != clusters or not 1 <= int(clusters) <= matrix.shape[0]:
        raise ValueError(f"clusters must be an integer from 1 to {matrix.shape[0]}")
    return matrix


def _recompute_centers(
    data: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    clusters: int,
) -> np.ndarray:
    new_centers = np.empty_like(centers)
    nearest_distance = np.min(squared_distances(data, centers), axis=1)
    available = np.argsort(nearest_distance)[::-1].tolist()
    for cluster in range(clusters):
        rows = data[labels == cluster]
        if rows.size:
            new_centers[cluster] = np.mean(rows, axis=0)
            continue
        fallback = next(index for index in available if labels[index] != cluster)
        available.remove(fallback)
        new_centers[cluster] = data[fallback]
    return new_centers


def _canonicalize_clusters(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort(tuple(centers[:, column] for column in reversed(range(centers.shape[1]))))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return inverse[labels], centers[order]


__all__ = ["kmeans_st"]

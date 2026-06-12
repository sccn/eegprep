"""Deterministic k-means numeric kernel shared by STUDY clustering helpers.

These functions hold the clustering numerics so that ``pop_clust``,
``optimal_kmeans``, and ``robust_kmeans`` all import downward from this module
instead of one user-facing wrapper. Labels are returned 1-based to match the
EEGLAB-facing cluster numbering convention used by the callers.
"""

from __future__ import annotations

import numpy as np


KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
KMEANS_TOLERANCE = 1e-8


def kmeans_labels(data: np.ndarray, clus_num: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic multi-restart k-means and return 1-based labels and centers."""
    rng = np.random.default_rng(random_state)
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    for _attempt in range(KMEANS_N_INIT):
        centers = data[rng.choice(data.shape[0], size=clus_num, replace=False)].copy()
        labels = np.zeros(data.shape[0], dtype=int)
        for _iteration in range(KMEANS_MAX_ITER):
            labels = np.argmin(squared_distances(data, centers), axis=1)
            new_centers = _recompute_centers(data, labels, centers, clus_num)
            if np.allclose(new_centers, centers, rtol=0, atol=KMEANS_TOLERANCE):
                centers = new_centers
                break
            centers = new_centers
        distances = squared_distances(data, centers)
        inertia = float(np.sum(distances[np.arange(data.shape[0]), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()
    if best_labels is None or best_centers is None:
        raise ValueError("K-means failed to initialize clusters")
    return best_labels.astype(int) + 1, best_centers


def squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return the matrix of squared Euclidean distances from rows to centers."""
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff * diff, axis=2)


def _recompute_centers(data: np.ndarray, labels: np.ndarray, centers: np.ndarray, clus_num: int) -> np.ndarray:
    new_centers = np.empty_like(centers)
    nearest_distance = np.min(squared_distances(data, centers), axis=1)
    fallback_index = int(np.argmax(nearest_distance))
    for cluster in range(clus_num):
        rows = data[labels == cluster]
        new_centers[cluster] = np.mean(rows, axis=0) if rows.size else data[fallback_index]
    return new_centers


__all__ = ["KMEANS_MAX_ITER", "KMEANS_N_INIT", "KMEANS_TOLERANCE", "kmeans_labels", "squared_distances"]

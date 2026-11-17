"""Hartigan-Wong k-means clustering for row-wise observations."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.eucl import eucl
from eegprep.functions.miscfunc.uniquef import uniquef


KMEANS_MAX_ITER = 300
KMEANS_TOLERANCE = 1e-8
KMEANS_ST_MAX_ITER = 50
_HIGHVAL = 10e8
_UPGMA_PLUG = 10e6


def kmeans_st(
    data: Any,
    clusters: int,
    restarts: int = 0,
    *,
    random_state: int = 0,
    max_iter: int = KMEANS_ST_MAX_ITER,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Partition rows with EEGLAB's Hartigan-Wong k-means.

    The first pass builds centroids from a UPGMA cut and does not draw random
    numbers. Restarts draw a NumPy permutation from ``random_state``, not
    MATLAB ``randperm``, so restart membership is reproducible here and not
    comparable to EEGLAB. Labels are zero-based indices of the centroid rows
    from the kept start.

    Args:
        data: Two-dimensional observation matrix.
        clusters: Number of clusters. At least two.
        restarts: Extra random starts after the UPGMA pass. A better solution
            resets the remaining count.
        random_state: Seed for restart permutations.
        max_iter: Maximum transfer passes per start.

    Returns:
        Centroids, labels, and total squared error.
    """
    matrix = _validated_observations(data, clusters)
    cluster_count = int(clusters)
    if cluster_count < 2:
        raise ValueError("clusters must be at least 2")
    if int(restarts) != restarts or restarts < 0:
        raise ValueError("restarts must be a non-negative integer")
    if int(max_iter) != max_iter or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    restart_count = int(restarts)
    restart_iter = restart_count + 1
    best_sse = _HIGHVAL
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    use_upgma = True
    rng = np.random.default_rng(int(random_state))
    while restart_iter > 0:
        restart_iter -= 1
        if use_upgma:
            use_upgma = False
            centers = _upgma_centers(matrix, cluster_count)
        else:
            centers = matrix[rng.permutation(matrix.shape[0])[:cluster_count]].copy()
        labels, centers, sse = _transfer_points(matrix, centers, int(max_iter))
        if sse < best_sse:
            best_sse = sse
            best_labels = labels.copy()
            best_centers = centers.copy()
            restart_iter = restart_count
    if best_labels is None or best_centers is None:
        raise ValueError("k-means failed to initialize clusters")
    return best_centers, best_labels, best_sse


def squared_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances from every row to every center."""
    differences = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(differences * differences, axis=2)


def _upgma_centers(data: np.ndarray, clusters: int) -> np.ndarray:
    count, width = data.shape
    topology = _upgma_topology(_pairwise_eucl(data))
    groups = np.arange(1, count + 1, dtype=float)
    for step in range(count - clusters):
        left, right, merged = topology[step, :3]
        groups[groups == left] = merged
        groups[groups == right] = merged
    group_ids, _counts, _first = uniquef(groups)
    centers = np.zeros((clusters, width), dtype=float)
    for index, group_id in enumerate(group_ids):
        rows = data[groups == group_id]
        centers[index] = np.mean(rows, axis=0) if rows.shape[0] > 1 else rows[0]
    return centers


def _upgma_topology(dist: np.ndarray) -> np.ndarray:
    dist = np.array(dist, dtype=float, copy=True)
    count = dist.shape[0]
    eps = np.finfo(float).eps
    if dist.shape != (count, count) or np.any(np.diag(dist) > eps):
        raise ValueError("UPGMA input matrix is not a distance matrix")
    sizes = np.ones(count, dtype=float)
    cluster_id = np.arange(1, count + 1, dtype=int)
    topology = np.zeros((count - 1, 4), dtype=float)
    np.fill_diagonal(dist, _UPGMA_PLUG)
    for step in range(1, count):
        min_dist = float(np.min(dist))
        pair = _first_upper_pair(dist, min_dist)
        left_id = int(cluster_id[pair[0]])
        right_id = int(cluster_id[pair[1]])
        merged = count + step
        if left_id < right_id:
            topology[step - 1] = (left_id, right_id, merged, min_dist)
        else:
            topology[step - 1] = (right_id, left_id, merged, min_dist)
        cluster_id[pair[0]] = merged
        dist[pair[0], pair[1]] = _UPGMA_PLUG
        dist[pair[1], pair[0]] = _UPGMA_PLUG
        new_size = sizes[pair[0]] + sizes[pair[1]]
        alpha_left = sizes[pair[0]] / new_size
        alpha_right = sizes[pair[1]] / new_size
        sizes[pair[0]] = new_size
        for other in range(count):
            if other in pair:
                continue
            dist[other, pair[0]] = alpha_left * dist[other, pair[0]] + alpha_right * dist[other, pair[1]]
            dist[pair[0], other] = alpha_left * dist[pair[0], other] + alpha_right * dist[pair[1], other]
            dist[other, pair[1]] = _UPGMA_PLUG
            dist[pair[1], other] = _UPGMA_PLUG
    return topology


def _first_upper_pair(dist: np.ndarray, min_dist: float) -> tuple[int, int]:
    count = dist.shape[0]
    for linear in np.flatnonzero(dist.ravel(order="F") == min_dist):
        row = int(linear % count)
        col = int(linear // count)
        if row <= col:
            return row, col
    raise RuntimeError("UPGMA found no upper-triangle minimum")


def _transfer_points(
    data: np.ndarray,
    centers: np.ndarray,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = data.shape[0]
    cluster_count = centers.shape[0]
    centers = centers.copy()
    distances = np.column_stack([_eucl_to_rows(data, centers[cluster]) for cluster in range(cluster_count)])
    nearest = np.argsort(distances, axis=1, kind="stable")
    own = nearest[:, 0].copy()
    second = nearest[:, 1].copy()
    counts = np.zeros(cluster_count, dtype=int)
    for cluster in range(cluster_count):
        rows = data[own == cluster]
        counts[cluster] = rows.shape[0]
        centers[cluster] = _column_means(rows, centers.shape[1])
    live = np.ones(cluster_count, dtype=int)
    eps = np.finfo(float).eps
    adj1 = counts / (counts + 1)
    adj2 = counts / (counts - 1 + eps)
    update = np.full(cluster_count, count, dtype=int)
    adjusted = np.array([_own_adjusted(data, centers, own, adj2, index) for index in range(count)])

    iteration = 0
    while iteration < max_iter and np.any(live):
        iteration += 1
        index = -1
        while index + 1 < count and np.any(live):
            index += 1
            update -= 1
            current = int(own[index])
            if counts[current] > 1:
                candidate = adj1[current] * _eucl_to_rows(centers, data[index]) ** 2
                if live[current]:
                    candidate[current] = _HIGHVAL
                else:
                    inactive = np.flatnonzero(live == 0)
                    candidate[np.unique(np.r_[current, inactive])] = _HIGHVAL
                best_value, destination = _matlab_min(candidate)
                if best_value >= adjusted[index]:
                    second[index] = destination
                else:
                    own[index] = destination
                    second[index] = current
                    for cluster in (current, destination):
                        rows = data[own == cluster]
                        if rows.shape[0] > 1:
                            centers[cluster] = np.mean(rows, axis=0)
                        else:
                            centers[cluster] = rows[0]
                        counts[cluster] = rows.shape[0]
                    moved = np.array([current, destination])
                    adj1[moved] = counts[moved] / (counts[moved] + 1)
                    adj2[moved] = counts[moved] / (counts[moved] - 1 + eps)
                    adjusted[index] = _own_adjusted(data, centers, own, adj2, index)
                    live[moved] = 1
                    update[moved] = count
            stale = np.flatnonzero(update < 1)
            if stale.size:
                live[stale] = 0
        if np.any(live):
            for index in range(count):
                current = int(own[index])
                destination = int(second[index])
                if np.any(live[[current, destination]]) and counts[current] > 1:
                    challenger = (
                        adj1[destination]
                        * _eucl_to_rows(
                            centers[destination : destination + 1],
                            data[index : index + 1],
                        )[0]
                        ** 2
                    )
                    # EEGLAB's if() on R1 is true only when every element is true.
                    if np.all(adjusted >= challenger):
                        own[index], second[index] = second[index], own[index]
                        for cluster in (current, destination):
                            rows = data[own == cluster]
                            centers[cluster] = _matlab_mean_vector(rows, centers.shape[1])
                            counts[cluster] = rows.shape[0]
                        moved = np.array([current, destination])
                        adj1[moved] = counts[moved] / (counts[moved] + 1)
                        adj2[moved] = counts[moved] / (counts[moved] - 1 + eps)
                        adjusted[index] = _own_adjusted(data, centers, own, adj2, index)
                        live[moved] = 1
                        update[moved] = count
    sse = 0.0
    for index in range(count):
        sse += float(_eucl_to_rows(data[index : index + 1], centers[own[index]])[0] ** 2)
    return own.astype(int), centers, float(sse)


def _pairwise_eucl(points: np.ndarray) -> np.ndarray:
    return np.asarray(eucl(points), dtype=float)


def _own_adjusted(
    data: np.ndarray,
    centers: np.ndarray,
    own: np.ndarray,
    adj2: np.ndarray,
    index: int,
) -> float:
    distance = _eucl_to_rows(data[index : index + 1], centers[own[index]])[0]
    return float(adj2[own[index]] * distance**2)


def _eucl_to_rows(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    values = eucl(np.asarray(points, dtype=float), np.reshape(np.asarray(reference, dtype=float), (1, -1)))
    return np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)


def _column_means(rows: np.ndarray, width: int) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.full(width, np.nan)
    return np.mean(rows, axis=0)


def _matlab_mean_vector(rows: np.ndarray, width: int) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.full(width, np.nan)
    if rows.shape[0] == 1:
        return np.full(width, float(np.mean(rows)))
    return np.mean(rows, axis=0)


def _matlab_min(values: np.ndarray) -> tuple[float, int]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float(values[0]), 0
    index = int(np.argmin(np.where(finite, values, np.inf)))
    return float(values[index]), index


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
    return best_labels, best_centers


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


__all__ = ["kmeans_st"]

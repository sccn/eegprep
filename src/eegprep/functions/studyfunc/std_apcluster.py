"""Affinity-propagation-style exemplar clustering for STUDY data."""

from __future__ import annotations

from typing import Any

import numpy as np


def std_apcluster(
    clustdata: Any,
    *,
    maxits: int = 200,
    convits: int = 100,
    dampfact: float = 0.9,
    dist: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster rows using deterministic affinity propagation updates."""
    data = np.asarray(clustdata, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("clustdata must be a 2-D matrix with at least two rows")
    if not 0.5 <= float(dampfact) < 1.0:
        raise ValueError("dampfact must be in [0.5, 1)")
    similarity = -_distance_matrix(data, dist)
    preference = float(np.median(similarity))
    np.fill_diagonal(similarity, preference)
    responsibilities = np.zeros_like(similarity)
    availabilities = np.zeros_like(similarity)
    previous_exemplars: tuple[int, ...] = ()
    stable = 0
    for _iteration in range(int(maxits)):
        old = responsibilities.copy()
        combined = availabilities + similarity
        best = np.max(combined, axis=1)
        best_index = np.argmax(combined, axis=1)
        combined[np.arange(data.shape[0]), best_index] = -np.inf
        second_best = np.max(combined, axis=1)
        responsibilities = similarity - best[:, np.newaxis]
        responsibilities[np.arange(data.shape[0]), best_index] = (
            similarity[np.arange(data.shape[0]), best_index] - second_best
        )
        responsibilities = float(dampfact) * old + (1.0 - float(dampfact)) * responsibilities

        old = availabilities.copy()
        positive = np.maximum(responsibilities, 0.0)
        positive[np.diag_indices_from(positive)] = responsibilities[np.diag_indices_from(responsibilities)]
        availabilities = np.minimum(
            0.0, responsibilities.diagonal()[np.newaxis, :] + np.sum(positive, axis=0) - positive
        )
        availabilities[np.diag_indices_from(availabilities)] = np.sum(
            np.maximum(responsibilities, 0.0), axis=0
        ) - np.maximum(responsibilities.diagonal(), 0.0)
        availabilities = float(dampfact) * old + (1.0 - float(dampfact)) * availabilities

        exemplars = tuple(np.flatnonzero((responsibilities + availabilities).diagonal() > 0.0).tolist())
        if exemplars == previous_exemplars:
            stable += 1
        else:
            stable = 0
            previous_exemplars = exemplars
        if stable >= int(convits):
            break
    exemplars = list(previous_exemplars) or [int(np.argmax((responsibilities + availabilities).diagonal()))]
    labels = _assign_exemplars(similarity, exemplars)
    centers = data[exemplars, :]
    sumd = _sum_distances(data, labels, centers)
    return labels, centers, sumd


def _distance_matrix(data: np.ndarray, metric: str) -> np.ndarray:
    metric = metric.lower()
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=2))
    if metric == "cityblock":
        return np.sum(np.abs(diff), axis=2)
    if metric == "cosine":
        norm = np.linalg.norm(data, axis=1)
        denom = np.maximum(norm[:, np.newaxis] * norm[np.newaxis, :], np.finfo(float).eps)
        return 1.0 - (data @ data.T) / denom
    raise ValueError("std_apcluster supports 'euclidean', 'cityblock', and 'cosine' distances")


def _assign_exemplars(similarity: np.ndarray, exemplars: list[int]) -> np.ndarray:
    selected = similarity[:, exemplars]
    raw = np.argmax(selected, axis=1)
    remap = {position: index + 1 for index, position in enumerate(range(len(exemplars)))}
    return np.asarray([remap[int(item)] for item in raw], dtype=int)


def _sum_distances(data: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    sums = []
    for label in sorted(set(labels.tolist())):
        rows = data[labels == label]
        center = centers[label - 1]
        sums.append(float(np.sum(np.linalg.norm(rows - center, axis=1))))
    return np.asarray(sums, dtype=float)


__all__ = ["std_apcluster"]

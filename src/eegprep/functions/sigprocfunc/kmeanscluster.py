"""Simple deterministic k-means clustering."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.kmeans_st import (
    _kmeans_labels,
    _validated_observations,
)


def kmeanscluster(
    data: Any,
    clusters: int = 1,
    randomized: bool = False,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster row-wise observations using EEGLAB's simple k-means contract.

    Returns zero-based labels, centroids, the converged previous-label vector,
    and an unchanged copy of the input observations. Deterministic mode
    initializes from the first ``clusters`` rows, as EEGLAB does.
    """
    matrix = _validated_observations(data, clusters)
    initial = None if randomized else matrix[: int(clusters)]
    labels, centers = _kmeans_labels(
        matrix,
        int(clusters),
        random_state=int(random_state),
        n_init=1,
        initial_centers=initial,
    )
    return labels, centers, labels.copy(), matrix.copy()


__all__ = ["kmeanscluster"]

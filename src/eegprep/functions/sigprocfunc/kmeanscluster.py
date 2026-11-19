"""Simple deterministic k-means clustering."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.kmeans_st import (
    _kmeans_labels,
    _validated_observations,
)
from eegprep.functions.miscfunc.matlab_outputs import as_column


def kmeanscluster(
    data: Any,
    clusters: int = 1,
    randomized: bool = False,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cluster row-wise observations using EEGLAB's simple k-means contract.

    Returns 1-based labels, centroids, the converged previous-label vector,
    and an unchanged copy of the input observations. Deterministic mode
    initializes from the first ``clusters`` rows, as EEGLAB does, and keeps
    that centroid-row numbering.
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
    return as_column(labels + 1), centers, as_column(labels + 1), matrix.copy()


__all__ = ["kmeanscluster"]

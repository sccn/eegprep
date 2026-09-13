"""Deterministic k-means numeric kernel shared by STUDY clustering helpers.

These functions hold the clustering numerics so that ``pop_clust``,
``optimal_kmeans``, and ``robust_kmeans`` all import downward from this module
instead of one user-facing wrapper. Labels are returned 1-based to match the
EEGLAB-facing cluster numbering convention used by the callers.
"""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.kmeans_st import _kmeans_labels, squared_distances


KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
KMEANS_TOLERANCE = 1e-8


def kmeans_labels(data: np.ndarray, clus_num: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic multi-restart k-means and return 1-based labels and centers."""
    labels, centers = _kmeans_labels(
        data,
        clus_num,
        random_state=random_state,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
    )
    return labels + 1, centers


__all__ = ["KMEANS_MAX_ITER", "KMEANS_N_INIT", "KMEANS_TOLERANCE", "kmeans_labels", "squared_distances"]

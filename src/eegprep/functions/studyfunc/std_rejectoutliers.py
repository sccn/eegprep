"""Reject STUDY cluster outliers by preclustering distance."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_at,
    cluster_command,
    cluster_list,
)
from eegprep.functions.studyfunc.std_moveoutlier import std_moveoutlier


def std_rejectoutliers(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    clusters: Any = "all",
    th: float = 3.0,
    *,
    return_com: bool = False,
) -> Any:
    """Move components farther than ``th`` cluster-distance stds to outliers."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    threshold = float(th)
    if threshold <= 0:
        raise ValueError("Outlier threshold must be greater than 0")
    selected = _selected_clusters(study, clusters)
    for cluster_index in selected:
        cluster = cluster_at(study, cluster_index)
        lower = str(cluster.get("name") or "").lower()
        if lower.startswith(("parentcluster", "notclust", "outlier")) or cluster.get("child"):
            continue
        data = np.asarray((cluster.get("preclust") or {}).get("preclustdata", []), dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            continue
        centroid = np.mean(data, axis=0, keepdims=True)
        distances = np.linalg.norm(data - centroid, axis=1)
        spread = float(np.std(distances))
        if spread == 0:
            continue
        outliers = np.flatnonzero(distances > spread * threshold) + 1
        if outliers.size:
            study = std_moveoutlier(study, datasets, cluster_index, outliers.tolist())
    command = cluster_command(
        "std_rejectoutliers",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        clusters=clusters,
        th=th,
    )
    return (study, command) if return_com else study


def _selected_clusters(study: dict[str, Any], clusters: Any) -> list[int]:
    if isinstance(clusters, str) and clusters.lower() == "all":
        return list(range(2, len(cluster_list(study)) + 1))
    values = np.asarray(clusters, dtype=int).ravel()
    if values.size == 0:
        return list(range(2, len(cluster_list(study)) + 1))
    return values.tolist()


__all__ = ["std_rejectoutliers"]

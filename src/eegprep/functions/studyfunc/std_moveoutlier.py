"""Move STUDY cluster components to an outlier cluster."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_at,
    cluster_command,
    cluster_list,
    next_cluster_number,
)
from eegprep.functions.studyfunc.std_movecomp import std_movecomp


OUTLIER_SOURCE_FIELD = "outlier_parent_cluster"


def std_moveoutlier(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    old_clus: int,
    comps: Any,
    *,
    return_com: bool = False,
) -> Any:
    """Move component positions from a cluster into its outlier cluster."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    cluster_index = int(old_clus)
    cluster = cluster_at(study, cluster_index)
    lower = str(cluster.get("name") or "").lower()
    if lower.startswith(("notclust", "outlier")):
        raise ValueError("Cannot move components from Notclust or Outliers clusters")
    if cluster.get("child"):
        raise ValueError("Cannot move components if cluster has child clusters")
    outlier_index = _find_outlier_cluster(study, cluster_index)
    if outlier_index == 0:
        study = _create_outlier_cluster(study, cluster_index)
        outlier_index = len(study["cluster"])
    study = std_movecomp(study, datasets, cluster_index, outlier_index, comps)
    command = cluster_command(
        "std_moveoutlier",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        str(cluster_index),
        comps=np.asarray(comps, dtype=int).ravel().tolist(),
    )
    return (study, command) if return_com else study


def _find_outlier_cluster(study: dict[str, Any], cluster_index: int) -> int:
    cluster_name = str(cluster_at(study, cluster_index).get("name") or "")
    for index, cluster in enumerate(cluster_list(study), start=1):
        if str(cluster.get(OUTLIER_SOURCE_FIELD) or "") == cluster_name:
            return index
    prefix = f"outliers {cluster_name} ".lower()
    for index, cluster in enumerate(cluster_list(study), start=1):
        if str(cluster.get("name") or "").lower().startswith(prefix):
            return index
    return 0


def _create_outlier_cluster(study: dict[str, Any], cluster_index: int) -> dict[str, Any]:
    study = deepcopy(study)
    clusters = cluster_list(study)
    source = clusters[cluster_index - 1]
    source_name = str(source.get("name") or "")
    name = f"Outliers {source_name} {next_cluster_number(clusters)}"
    parent_names = list(source.get("parent") or [])
    clusters.append(
        {
            "name": name,
            "sets": [],
            "comps": [],
            "parent": parent_names,
            "child": [],
            "algorithm": ["manual_outlier"],
            "preclust": {
                "preclustparams": deepcopy((source.get("preclust") or {}).get("preclustparams", [])),
                "preclustdata": [],
            },
            OUTLIER_SOURCE_FIELD: source_name,
        }
    )
    for parent in parent_names:
        for entry in clusters:
            if str(entry.get("name") or "") == parent and name not in entry.get("child", []):
                entry.setdefault("child", []).append(name)
                break
    study["cluster"] = clusters
    return study


__all__ = ["std_moveoutlier"]

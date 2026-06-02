"""Rename STUDY component clusters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_at,
    cluster_command,
    cluster_list,
    trailing_cluster_number,
)
from eegprep.functions.studyfunc.std_moveoutlier import OUTLIER_SOURCE_FIELD


def std_renameclust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    cls: int,
    new_name: str,
    *,
    return_com: bool = False,
) -> Any:
    """Rename one 1-based STUDY cluster and update parent/child links."""
    study, _datasets = checked_study_and_datasets(STUDY, ALLEEG)
    cluster_index = int(cls)
    cluster = cluster_at(study, cluster_index)
    if str(cluster.get("name") or "").lower().startswith("notclust"):
        raise ValueError("Notclust clusters cannot be renamed")
    suffix = trailing_cluster_number(str(cluster.get("name") or ""))
    renamed = f"{new_name} {suffix}".strip()
    old_name = str(cluster.get("name") or "")
    clusters = cluster_list(study)
    outlier_index = _find_outlier_cluster(clusters, old_name)
    clusters[cluster_index - 1]["name"] = renamed
    for entry in clusters:
        entry["parent"] = [renamed if parent == old_name else parent for parent in entry.get("parent", [])]
        entry["child"] = [renamed if child == old_name else child for child in entry.get("child", [])]
    if outlier_index:
        outlier = clusters[outlier_index - 1]
        old_outlier_name = str(outlier.get("name") or "")
        outlier_suffix = trailing_cluster_number(old_outlier_name)
        new_outlier_name = f"Outliers {renamed} {outlier_suffix}".strip()
        outlier["name"] = new_outlier_name
        outlier[OUTLIER_SOURCE_FIELD] = renamed
        for entry in clusters:
            entry["parent"] = [
                new_outlier_name if parent == old_outlier_name else parent for parent in entry.get("parent", [])
            ]
            entry["child"] = [
                new_outlier_name if child == old_outlier_name else child for child in entry.get("child", [])
            ]
    study = deepcopy(study)
    study["cluster"] = clusters
    study["saved"] = "no"
    command = cluster_command(
        "std_renameclust",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        str(cluster_index),
        new_name=new_name,
    )
    return (study, command) if return_com else study


def _find_outlier_cluster(clusters: list[dict[str, Any]], cluster_name: str) -> int:
    for index, cluster in enumerate(clusters, start=1):
        if str(cluster.get(OUTLIER_SOURCE_FIELD) or "") == cluster_name:
            return index
    prefix = f"outliers {cluster_name} ".lower()
    for index, cluster in enumerate(clusters, start=1):
        if str(cluster.get("name") or "").lower().startswith(prefix):
            return index
    return 0


__all__ = ["std_renameclust"]

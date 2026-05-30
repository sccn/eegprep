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
    clusters[cluster_index - 1]["name"] = renamed
    for entry in clusters:
        entry["parent"] = [renamed if parent == old_name else parent for parent in entry.get("parent", [])]
        entry["child"] = [renamed if child == old_name else child for child in entry.get("child", [])]
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


__all__ = ["std_renameclust"]

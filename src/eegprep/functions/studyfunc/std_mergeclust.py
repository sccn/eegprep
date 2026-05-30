"""Merge STUDY component clusters."""

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
    sets_array,
)


def std_mergeclust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    mrg_cls: Any,
    name: str = "Cls",
    *,
    return_com: bool = False,
) -> Any:
    """Merge two or more 1-based STUDY clusters into a new child cluster."""
    study, _datasets = checked_study_and_datasets(STUDY, ALLEEG)
    merge_indices = np.asarray(mrg_cls, dtype=int).ravel()
    if merge_indices.size < 2:
        raise ValueError("std_mergeclust requires at least two clusters")
    clusters = cluster_list(study)
    merged_sets = []
    merged_comps = []
    merged_data = []
    parents = []
    for index in merge_indices:
        cluster = cluster_at(study, int(index))
        lower = str(cluster.get("name") or "").lower()
        if lower.startswith(("notclust", "outlier")) or cluster.get("child"):
            raise ValueError("Cannot merge Notclust, Outliers, or parent clusters")
        parents.append(str(cluster.get("name") or ""))
        merged_sets.append(sets_array(cluster.get("sets")).astype(int))
        merged_comps.append(np.asarray(cluster.get("comps") or [], dtype=int))
        data = np.asarray((cluster.get("preclust") or {}).get("preclustdata", []), dtype=float)
        if data.size:
            merged_data.append(data)
    sets = np.concatenate(merged_sets, axis=1)
    comps = np.concatenate(merged_comps)
    order = np.lexsort((comps, sets[0]))
    new_number = next_cluster_number(clusters)
    entry: dict[str, Any] = {
        "name": f"{name} {new_number}",
        "sets": sets[:, order].tolist(),
        "comps": comps[order].tolist(),
        "parent": parents,
        "child": [],
        "algorithm": ["manual_merge", merge_indices.tolist()],
        "preclust": {"preclustparams": [], "preclustdata": []},
    }
    if merged_data:
        entry["preclust"] = {
            "preclustparams": deepcopy(
                (cluster_at(study, int(merge_indices[0])).get("preclust") or {}).get("preclustparams", [])
            ),
            "preclustdata": np.concatenate(merged_data, axis=0)[order, :].tolist(),
        }
    clusters.append(entry)
    for index in merge_indices:
        clusters[int(index) - 1].setdefault("child", []).append(entry["name"])
    study = deepcopy(study)
    study["cluster"] = clusters
    study["saved"] = "no"
    command = cluster_command(
        "std_mergeclust",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        mrg_cls=merge_indices.tolist(),
        name=name,
    )
    return (study, command) if return_com else study


__all__ = ["std_mergeclust"]

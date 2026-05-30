"""Create EEGLAB-style STUDY component clusters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_command,
    cluster_list,
    ensure_parent_cluster,
    next_cluster_number,
    remove_child_clusters,
    rows_for_cluster,
)


def std_createclust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    *,
    clusterind: Any,
    algorithm: Any = None,
    name: str = "Cls",
    parentcluster: str = "off",
    ignore0: str = "off",
    return_com: bool = False,
) -> Any:
    """Create child clusters from one cluster label per preclustering row."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    study = ensure_parent_cluster(study, datasets)
    labels = np.asarray(clusterind, dtype=int).ravel()
    if labels.size == 0:
        raise ValueError("std_createclust requires clusterind labels")
    parent_index = 1 if parentcluster == "on" else int(study.get("etc", {}).get("preclust", {}).get("clustlevel", 1))
    sets, comps = rows_for_cluster(study, datasets, parent_index)
    if labels.size != comps.size:
        raise ValueError("clusterind length must match the selected parent cluster components")

    study = remove_child_clusters(study, parent_index)
    clusters = cluster_list(study)
    parent = clusters[parent_index - 1]
    parent_name = str(parent.get("name") or "ParentCluster")
    next_number = next_cluster_number(clusters)
    labels_to_make = [label for label in sorted(set(labels.tolist())) if label > 0 or ignore0 == "off"]
    preclustdata = np.asarray(study.get("etc", {}).get("preclust", {}).get("preclustdata", []), dtype=float)
    preclustparams = study.get("etc", {}).get("preclust", {}).get("preclustparams", [])
    parent["child"] = []

    for label in labels_to_make:
        selected = np.flatnonzero(labels == label)
        if selected.size == 0:
            continue
        cluster_name = f"outlier {next_number}" if label == 0 else f"{name} {next_number}"
        entry: dict[str, Any] = {
            "name": cluster_name,
            "sets": sets[:, selected].astype(int).tolist(),
            "comps": comps[selected].astype(int).tolist(),
            "parent": [parent_name],
            "child": [],
            "algorithm": algorithm or [],
            "preclust": {
                "preclustparams": deepcopy(preclustparams),
                "preclustdata": preclustdata[selected, :].tolist() if preclustdata.size else [],
            },
        }
        clusters.append(entry)
        parent["child"].append(cluster_name)
        next_number += 1

    clusters[parent_index - 1] = parent
    study = deepcopy(study)
    study["cluster"] = clusters
    study["saved"] = "no"
    command = cluster_command(
        "std_createclust",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        clusterind=labels.tolist(),
        algorithm=algorithm,
        name=name,
        parentcluster=parentcluster,
        ignore0=ignore0,
    )
    return (study, command) if return_com else study


__all__ = ["std_createclust"]

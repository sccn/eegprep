"""Shared helpers for EEGLAB-style STUDY component clustering."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import component_maps, python_literal
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, ensure_study, sync_datasetinfo


def checked_study_and_datasets(
    STUDY: dict[str, Any] | None, ALLEEG: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a STUDY with datasetinfo synchronized to loaded datasets."""
    datasets = as_alleeg_list(ALLEEG)
    if not datasets:
        raise ValueError("STUDY clustering requires loaded ALLEEG datasets")
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    return study, datasets


def ensure_parent_cluster(study: dict[str, Any], datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Ensure ``STUDY.cluster(1)`` contains all selected ICA components."""
    study = deepcopy(study)
    clusters = cluster_list(study)
    parent = clusters[0] if clusters else {}
    if parent.get("comps") and parent.get("sets"):
        clusters[0] = normalize_cluster(parent)
        study["cluster"] = clusters
        return study

    sets: list[int] = []
    comps: list[int] = []
    for study_set, info in enumerate(study.get("datasetinfo") or [], start=1):
        eeg = dataset_for_study_set(study, datasets, study_set)
        selected = list(info.get("comps") or [])
        if not selected:
            weights = np.asarray(eeg.get("icaweights", []), dtype=float)
            if weights.size:
                selected = list(range(1, int(weights.shape[0]) + 1))
        if not selected:
            raise ValueError(f"Dataset {study_set} has no ICA components for STUDY clustering")
        for comp in selected:
            sets.append(study_set)
            comps.append(int(comp))

    clusters = [
        {
            "name": str(parent.get("name") or "ParentCluster"),
            "sets": [sets],
            "comps": comps,
            "parent": [],
            "child": [],
            "preclust": parent.get("preclust") or {"preclustparams": [], "preclustdata": []},
        }
    ] + [normalize_cluster(cluster) for cluster in clusters[1:]]
    study["cluster"] = clusters
    return study


def cluster_list(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized STUDY cluster dictionaries."""
    value = study.get("cluster")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not value:
        value = [{"name": "ParentCluster", "sets": [], "comps": [], "child": [], "parent": []}]
    return [normalize_cluster(cluster) for cluster in value if isinstance(cluster, dict)]


def normalize_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    """Return one cluster with list-shaped EEGLAB fields."""
    output = deepcopy(cluster)
    output.setdefault("name", "")
    output.setdefault("sets", [])
    output.setdefault("comps", [])
    output.setdefault("parent", [])
    output.setdefault("child", [])
    output["sets"] = sets_array(output.get("sets")).astype(int).tolist()
    output["comps"] = [int(item) for item in np.asarray(output.get("comps") or [], dtype=int).ravel().tolist()]
    output["parent"] = _string_list(output.get("parent"))
    output["child"] = _string_list(output.get("child"))
    output.setdefault("preclust", {"preclustparams": [], "preclustdata": []})
    return output


def cluster_at(study: dict[str, Any], cluster_index: int) -> dict[str, Any]:
    """Return a 1-based cluster entry."""
    clusters = cluster_list(study)
    if cluster_index < 1 or cluster_index > len(clusters):
        raise ValueError(f"cluster index must be 1-based and within 1..{len(clusters)}")
    return clusters[cluster_index - 1]


def sets_array(value: Any) -> np.ndarray:
    """Return EEGLAB cluster sets as ``rows x components``."""
    if value is None:
        return np.empty((1, 0), dtype=int)
    if isinstance(value, np.ndarray) and value.size == 0:
        return np.empty((1, 0), dtype=int)
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return np.empty((1, 0), dtype=int)
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def dataset_for_study_set(study: dict[str, Any], datasets: list[dict[str, Any]], study_set: int) -> dict[str, Any]:
    """Return the ALLEEG dataset for a 1-based STUDY datasetinfo index."""
    infos = study.get("datasetinfo") or []
    if study_set < 1 or study_set > len(infos):
        raise ValueError(f"cluster set index {study_set} is outside STUDY.datasetinfo")
    dataset_index = int(infos[study_set - 1].get("index") or study_set)
    if dataset_index < 1 or dataset_index > len(datasets):
        raise ValueError(f"STUDY.datasetinfo({study_set}).index is outside ALLEEG")
    return datasets[dataset_index - 1]


def rows_for_cluster(
    study: dict[str, Any], datasets: list[dict[str, Any]], cluster_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster row sets and component numbers."""
    study = ensure_parent_cluster(study, datasets)
    cluster = cluster_at(study, cluster_index)
    sets = sets_array(cluster.get("sets"))
    comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
    if sets.shape[1] != comps.size:
        raise ValueError("STUDY.cluster sets and comps lengths do not match")
    if comps.size == 0:
        raise ValueError("Selected STUDY cluster contains no components")
    return sets, comps


def next_cluster_number(clusters: list[dict[str, Any]]) -> int:
    """Return the next trailing numeric cluster suffix."""
    highest = 0
    for cluster in clusters:
        match = re.search(r"(\d+)\s*$", str(cluster.get("name") or ""))
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def trailing_cluster_number(name: str) -> str:
    """Return a cluster name's trailing number or an empty string."""
    match = re.search(r"(\d+)\s*$", str(name or ""))
    return match.group(1) if match else ""


def cluster_command(function_name: str, targets: tuple[str, ...], *args: str, **kwargs: Any) -> str:
    """Build a pasteable Python STUDY history command."""
    pieces = list(args)
    pieces.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if not _empty_history(value))
    return f"{', '.join(targets)} = {function_name}({', '.join(pieces)})"


def remove_child_clusters(study: dict[str, Any], parent_index: int) -> dict[str, Any]:
    """Remove non-outlier child clusters below the selected clustering level."""
    study = deepcopy(study)
    clusters = cluster_list(study)
    parent_name = clusters[parent_index - 1].get("name")
    if parent_index == 1:
        clusters = [clusters[0]]
    else:
        clusters = [
            cluster
            for index, cluster in enumerate(clusters, start=1)
            if index == parent_index
            or parent_name not in _string_list(cluster.get("parent"))
            or str(cluster.get("name") or "").lower().startswith(("notclust", "outlier"))
        ]
    clusters[parent_index - 1]["child"] = []
    study["cluster"] = clusters
    return study


def component_label(study: dict[str, Any], sets: np.ndarray, comps: np.ndarray, row: int) -> str:
    """Return an EEGLAB-like component label for cluster dialogs."""
    study_set = int(sets[0, row])
    info = (study.get("datasetinfo") or [{}])[study_set - 1]
    subject = str(info.get("subject") or f"S{study_set}")
    return f"{subject}, IC {int(comps[row])}"


def cluster_summary(study: dict[str, Any]) -> list[str]:
    """Return human-readable cluster rows."""
    rows = []
    for index, cluster in enumerate(cluster_list(study), start=1):
        rows.append(f"{index}: {cluster.get('name') or ''} ({len(cluster.get('comps') or [])} ICs)")
    return rows


def _string_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _empty_history(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple, dict, set)) and len(value) == 0


__all__ = [
    "checked_study_and_datasets",
    "cluster_at",
    "cluster_command",
    "cluster_list",
    "cluster_summary",
    "component_label",
    "component_maps",
    "dataset_for_study_set",
    "ensure_parent_cluster",
    "next_cluster_number",
    "normalize_cluster",
    "remove_child_clusters",
    "rows_for_cluster",
    "sets_array",
    "trailing_cluster_number",
]

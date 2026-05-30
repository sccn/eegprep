"""Move STUDY components between clusters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_at,
    cluster_command,
    cluster_list,
    sets_array,
)


def std_movecomp(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    old_clus: int,
    new_clus: int,
    comps: Any,
    *,
    return_com: bool = False,
) -> Any:
    """Move component positions from one 1-based STUDY cluster to another."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    clusters = cluster_list(study)
    old_index = int(old_clus)
    new_index = int(new_clus)
    old = cluster_at(study, old_index)
    new = cluster_at(study, new_index)
    if old_index == 1 or new_index == 1:
        raise ValueError("Cannot move components to or from the ParentCluster")
    if old.get("child") or new.get("child"):
        raise ValueError("Cannot move components if clusters have child clusters")
    positions = np.asarray(comps, dtype=int).ravel()
    if positions.size == 0:
        raise ValueError("std_movecomp requires component positions to move")
    old_comps = np.asarray(old.get("comps") or [], dtype=int)
    old_sets = sets_array(old.get("sets")).astype(int)
    if np.any(positions < 1) or np.any(positions > old_comps.size):
        raise ValueError("component positions are outside the source cluster")
    selected = positions - 1

    new_comps = np.concatenate([np.asarray(new.get("comps") or [], dtype=int), old_comps[selected]])
    new_sets = np.concatenate([sets_array(new.get("sets")).astype(int), old_sets[:, selected]], axis=1)
    new_preclust = _append_preclust_rows(new, old, selected)
    keep = np.ones(old_comps.size, dtype=bool)
    keep[selected] = False
    old["comps"] = old_comps[keep].tolist()
    old["sets"] = old_sets[:, keep].tolist()
    old["preclust"] = _keep_preclust_rows(old, keep)
    order = np.lexsort((new_comps, new_sets[0]))
    new["comps"] = new_comps[order].tolist()
    new["sets"] = new_sets[:, order].tolist()
    new["preclust"] = _reorder_preclust_rows(new_preclust, order)

    clusters[old_index - 1] = old
    clusters[new_index - 1] = new
    study = deepcopy(study)
    study["cluster"] = clusters
    study["saved"] = "no"
    command = cluster_command(
        "std_movecomp",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        str(old_index),
        str(new_index),
        comps=positions.tolist(),
    )
    return (study, command) if return_com else study


def _append_preclust_rows(new: dict[str, Any], old: dict[str, Any], selected: np.ndarray) -> dict[str, Any]:
    new_preclust = deepcopy(new.get("preclust") or {})
    old_data = np.asarray((old.get("preclust") or {}).get("preclustdata", []), dtype=float)
    if old_data.size == 0:
        return new_preclust
    new_data = np.asarray(new_preclust.get("preclustdata", []), dtype=float)
    if new_data.size == 0:
        merged = old_data[selected, :]
    else:
        merged = np.concatenate([new_data, old_data[selected, :]], axis=0)
    new_preclust["preclustdata"] = merged.tolist()
    if not new_preclust.get("preclustparams"):
        new_preclust["preclustparams"] = deepcopy((old.get("preclust") or {}).get("preclustparams", []))
    return new_preclust


def _keep_preclust_rows(cluster: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    preclust = deepcopy(cluster.get("preclust") or {})
    data = np.asarray(preclust.get("preclustdata", []), dtype=float)
    if data.size:
        preclust["preclustdata"] = data[keep, :].tolist()
    return preclust


def _reorder_preclust_rows(preclust: dict[str, Any], order: np.ndarray) -> dict[str, Any]:
    output = deepcopy(preclust)
    data = np.asarray(output.get("preclustdata", []), dtype=float)
    if data.size:
        output["preclustdata"] = data[order, :].tolist()
    return output


__all__ = ["std_movecomp"]

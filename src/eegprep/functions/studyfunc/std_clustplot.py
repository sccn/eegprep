"""Lightweight STUDY cluster plotting hooks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.studyfunc._cluster_utils import checked_study_and_datasets, cluster_at, cluster_list


def std_clustplot(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    clusters: Any = None,
    *,
    measure: str = "preclust",
    return_com: bool = False,
) -> Any:
    """Plot a compact cluster summary for GUI and console workflows."""
    study, _datasets = checked_study_and_datasets(STUDY, ALLEEG)
    cluster_indices = _cluster_indices(study, clusters)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    names = []
    counts = []
    for index in cluster_indices:
        cluster = cluster_at(study, index)
        lower = str(cluster.get("name") or "").lower()
        if lower.startswith("parentcluster"):
            continue
        names.append(str(cluster.get("name") or f"Cluster {index}"))
        counts.append(len(cluster.get("comps") or []))
    if not names:
        names = ["No clusters"]
        counts = [0]
    ax.bar(np.arange(len(names)), counts, color="#5b8cc0")
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("IC count")
    ax.set_title(f"{study.get('name') or 'STUDY'} cluster {measure} summary")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    command = f"fig = std_clustplot(STUDY, ALLEEG, clusters={cluster_indices}, measure={measure!r})"
    return (study, command, fig) if return_com else fig


def _cluster_indices(study: dict[str, Any], clusters: Any) -> list[int]:
    if clusters is None or clusters == []:
        return list(range(1, len(cluster_list(study)) + 1))
    return np.asarray(clusters, dtype=int).ravel().tolist()


__all__ = ["std_clustplot"]

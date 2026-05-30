"""Cluster STUDY ICA components from a preclustering matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.studyfunc._cluster_utils import checked_study_and_datasets, cluster_command
from eegprep.functions.studyfunc.std_createclust import std_createclust


ALGORITHMS = ("kmeans", "kmeanscluster")


def pop_clust(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    *args: Any,
    algorithm: str = "kmeans",
    clus_num: int = 20,
    outliers: float = float("inf"),
    random_state: int = 0,
    gui: bool = False,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Run deterministic k-means clustering on ``STUDY.etc.preclust``."""
    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    if args:
        if len(args) % 2:
            raise ValueError("pop_clust key/value arguments must be in pairs")
        kwargs.update({str(args[index]).lower(): args[index + 1] for index in range(0, len(args), 2)})
    algorithm = str(kwargs.pop("algorithm", algorithm)).lower()
    clus_num = int(kwargs.pop("clus_num", clus_num))
    outliers = float(kwargs.pop("outliers", outliers))
    random_state = int(kwargs.pop("random_state", random_state))
    gui = bool(kwargs.pop("gui", gui))
    if kwargs:
        raise ValueError(f"Unknown pop_clust option(s): {', '.join(sorted(kwargs))}")
    if gui:
        result = inputgui(pop_clust_dialog_spec(study), renderer=renderer)
        if result is None:
            return (study, "") if return_com else study
        algorithm = _algorithm_from_gui(result.get("algorithm"))
        clus_num = _optional_int(result.get("clus_num"), _default_cluster_count(study))
        outliers = _optional_float(result.get("outliers"), 3.0) if bool(result.get("outliers_on")) else float("inf")

    preclust = study.get("etc", {}).get("preclust")
    if not isinstance(preclust, dict) or not preclust.get("preclustdata"):
        raise ValueError("No pre-clustering information, pre-cluster first")
    data = np.asarray(preclust["preclustdata"], dtype=float)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("Preclustering data must contain at least two components")
    if algorithm not in ALGORITHMS:
        raise NotImplementedError("pop_clust currently supports kmeans and kmeanscluster")
    if clus_num < 2:
        raise ValueError("Number of clusters must be at least 2")
    if clus_num > data.shape[0]:
        raise ValueError("Number of clusters cannot exceed the number of preclustered components")

    labels, centers = _kmeans_labels(data, clus_num, random_state)
    if np.isfinite(outliers):
        if outliers <= 0:
            raise ValueError("Outlier threshold must be greater than 0")
        labels = _mark_outliers(data, labels, centers, outliers)
    study = std_createclust(
        study,
        datasets,
        clusterind=labels,
        algorithm=["Kmeans", clus_num],
        name="Cls",
    )
    command = cluster_command(
        "pop_clust",
        ("STUDY",),
        "STUDY",
        "ALLEEG",
        algorithm=algorithm,
        clus_num=clus_num,
        outliers=outliers,
        random_state=random_state,
    )
    return (study, command) if return_com else study


def pop_clust_dialog_spec(STUDY: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like clustering algorithm dialog spec."""
    default_clusters = _default_cluster_count(STUDY)
    return DialogSpec(
        title="Set clustering algorithm -- pop_clust()",
        controls=(
            ControlSpec("text", "Performing clustering on cluster 'ParentCluster'", font_weight="bold"),
            ControlSpec("text", "Clustering algorithm:"),
            ControlSpec("popupmenu", "Kmeans (stat. toolbox)|Kmeanscluster (no toolbox)", tag="algorithm", value=1),
            ControlSpec("text", "Number of clusters to compute:"),
            ControlSpec("edit", tag="clus_num", value=str(default_clusters)),
            ControlSpec("checkbox", "Separate outliers (enter std.)", tag="outliers_on", value=0),
            ControlSpec("edit", tag="outliers", value="3", enabled=False),
        ),
        geometry=((1,), (1, 1.2), (1, 0.7), (1, 0.7)),
        function_name="pop_clust",
        eeglab_source="functions/studyfunc/pop_clust.m",
        help_text="pop_clust",
        size=(578, 282),
        content_margins=(38, 36, 38, 16),
        row_spacing=5,
        geomvert=(1, 0.5, 1, 1, 1),
        button_size=(58, 20),
        extra_stylesheet="""
            QDialog#pop_clust QLabel,
            QDialog#pop_clust QCheckBox,
            QDialog#pop_clust QPushButton,
            QDialog#pop_clust QLineEdit,
            QDialog#pop_clust QComboBox {
                font-size: 13px;
            }
            QDialog#pop_clust QLineEdit,
            QDialog#pop_clust QComboBox,
            QDialog#pop_clust QPushButton {
                min-height: 20px;
                max-height: 20px;
            }
        """,
    )


def _kmeans_labels(data: np.ndarray, clus_num: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=clus_num, n_init=10, random_state=random_state)
    labels = model.fit_predict(data).astype(int) + 1
    return labels, np.asarray(model.cluster_centers_, dtype=float)


def _mark_outliers(data: np.ndarray, labels: np.ndarray, centers: np.ndarray, threshold: float) -> np.ndarray:
    output = labels.copy()
    cluster_distances = []
    for label in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == label)
        if rows.size:
            cluster_distances.append(np.linalg.norm(data[rows] - centers[label - 1], axis=1))
    if not cluster_distances:
        return output
    reference_distance = float(np.mean([np.mean(distances) for distances in cluster_distances if distances.size]))
    for label, distances in zip(sorted(set(labels.tolist())), cluster_distances):
        rows = np.flatnonzero(labels == label)
        distances = np.linalg.norm(data[rows] - centers[label - 1], axis=1)
        spread = float(np.std(distances))
        if spread == 0:
            continue
        outlier_mask = (distances > spread * threshold) & (distances > reference_distance * threshold)
        output[rows[outlier_mask]] = 0
    return output


def _algorithm_from_gui(value: Any) -> str:
    try:
        index = int(value) - 1
    except (TypeError, ValueError):
        index = 0
    return ALGORITHMS[index] if 0 <= index < len(ALGORITHMS) else "kmeans"


def _default_cluster_count(study: dict[str, Any]) -> int:
    preclust = study.get("etc", {}).get("preclust", {})
    rows = len(preclust.get("preclustdata") or [])
    return max(2, min(20, int(np.ceil(rows / 2)) if rows else 2))


def _optional_int(value: Any, default: int) -> int:
    values = numeric_vector(value, dtype=int)
    return int(values[0]) if values.size else int(default)


def _optional_float(value: Any, default: float) -> float:
    values = numeric_vector(value, dtype=float)
    return float(values[0]) if values.size else float(default)


__all__ = ["pop_clust", "pop_clust_dialog_spec"]

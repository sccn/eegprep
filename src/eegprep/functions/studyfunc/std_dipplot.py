"""Plot already-localized DIPFIT components from STUDY clusters."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._cluster_utils import (
    checked_study_and_datasets,
    cluster_at,
    cluster_list,
    dataset_for_study_set,
    sets_array,
)
from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.plugins.dipfit._utils import normalize_model_list, one_based_indices


_VALID_MODES = {"apart", "together", "multicolor", "joined", "centroid", "comps"}


def std_dipplot(
    STUDY: dict[str, Any] | None,
    ALLEEG: Any,
    *args: Any,
    clusters: Any = None,
    comps: Any = None,
    mode: str | None = None,
    dipcolor: Any = None,
    dipsize: Any = None,
    plot: bool = True,
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Select and plot localized components from one or more STUDY clusters.

    This standalone workflow consumes existing ``EEG["dipfit"]["model"]``
    entries; it does not run source localization. Cluster, dataset, and
    component indices are EEGLAB-facing and therefore 1-based. ``comps``
    selects positions within each cluster's component list, matching EEGLAB's
    ``std_plotcompdip`` convention rather than selecting component numbers.

    The returned selection dictionaries expose ``dipoles`` and ``centroid``.
    Each dipole records its study-set index, ALLEEG index, component number,
    subject, ``posxyz``, ``momxyz``, and residual variance ``rv``. Modes
    ``joined`` and ``centroid`` implement the intent of the current EEGLAB test
    calls, which the pinned MATLAB implementation otherwise silently ignores.

    Args:
        STUDY: STUDY dictionary containing cluster membership.
        ALLEEG: Loaded EEG dataset dictionaries with DIPFIT models.
        clusters: 1-based cluster indices or ``"all"``. The parent cluster is
            excluded from ``"all"`` as in EEGLAB.
        comps: Optional 1-based positions within each selected cluster.
        mode: ``"apart"``, ``"together"``, ``"multicolor"``, ``"joined"``,
            ``"centroid"``, or ``"comps"``.
        dipcolor: Optional color or one color per selected cluster.
        dipsize: Optional marker size or one size per selected cluster.
        plot: Whether to create matplotlib figures.
        return_com: Append a replayable Python command to the result.

    Returns:
        ``(STUDY, selections, figures)`` and, when requested, the command.
        ``STUDY.cluster[*].dipole`` is updated with each selected centroid.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    clusters = options.pop("clusters", clusters)
    comps = options.pop("comps", comps)
    mode = options.pop("mode", mode)
    dipcolor = options.pop("dipcolor", dipcolor)
    dipsize = options.pop("dipsize", dipsize)
    plot = is_on(options.pop("plot", plot))
    ignored = {"plotsubjects"}
    unsupported = sorted(key for key in options if key not in ignored)
    if unsupported:
        raise ValueError(f"Unknown std_dipplot option(s): {', '.join(unsupported)}")

    study, datasets = checked_study_and_datasets(STUDY, ALLEEG)
    cluster_indices = _cluster_indices(study, clusters)
    selected_mode = _mode(mode, len(cluster_indices))
    selections = [_select_cluster(study, datasets, index, comps) for index in cluster_indices]
    _validate_coordinate_formats(selections)
    for selection in selections:
        study["cluster"][selection["cluster_index"] - 1]["dipole"] = deepcopy(selection["centroid"])

    colors = _plot_values(dipcolor, len(selections), "tab10")
    sizes = _plot_values(dipsize, len(selections), 55.0)
    figures = _plot_selections(selections, selected_mode, colors, sizes) if plot else []
    command = build_python_call(
        ("STUDY", "DIPOLES", "FIGURES"),
        "std_dipplot",
        "STUDY",
        "ALLEEG",
        clusters=clusters,
        comps=comps,
        mode=selected_mode,
        dipcolor=dipcolor,
        dipsize=dipsize,
        plot=plot if not plot else None,
    )
    result = (study, selections, figures)
    return (*result, command) if return_com else result


def _cluster_indices(study: dict[str, Any], clusters: Any) -> list[int]:
    all_clusters = cluster_list(study)
    if isinstance(clusters, str):
        if clusters.lower() != "all":
            raise ValueError("clusters must be 1-based indices or 'all'")
        indices = list(range(2, len(all_clusters) + 1))
    elif clusters is None:
        indices = [
            index
            for index, cluster in enumerate(all_clusters[1:], start=2)
            if not str(cluster.get("name") or "").lower().startswith(("notclust", "parentcluster"))
        ]
    elif _is_empty(clusters):
        indices = list(range(2, len(all_clusters) + 1))
    else:
        indices = one_based_indices(clusters, limit=len(all_clusters))
    if not indices:
        raise ValueError("No STUDY clusters are available to plot")
    return indices


def _mode(mode: str | None, cluster_count: int) -> str:
    value = str(mode or ("apart" if cluster_count == 1 else "together")).strip().lower()
    if value not in _VALID_MODES:
        valid = ", ".join(sorted(_VALID_MODES))
        raise ValueError(f"mode must be one of {valid}")
    return value


def _select_cluster(
    study: dict[str, Any], datasets: list[dict[str, Any]], cluster_index: int, comps: Any
) -> dict[str, Any]:
    cluster = cluster_at(study, cluster_index)
    component_numbers = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
    study_sets = sets_array(cluster.get("sets"))
    if study_sets.shape[1] != component_numbers.size:
        raise ValueError(f"STUDY cluster {cluster_index} sets and comps lengths do not match")
    selected_members = None if isinstance(comps, str) and comps.strip().lower() == "all" else comps
    member_indices = one_based_indices(selected_members, limit=component_numbers.size, default_all=True)
    dipoles = []
    for member_index in member_indices:
        study_set = int(study_sets[0, member_index - 1])
        component = int(component_numbers[member_index - 1])
        dataset = dataset_for_study_set(study, datasets, study_set)
        models = normalize_model_list(dataset)
        if component < 1 or component > len(models):
            raise ValueError(
                f"STUDY cluster {cluster_index} component {component} is outside dataset {study_set} DIPFIT models"
            )
        normalized = _dipole_model(models[component - 1], cluster_index, component)
        if normalized is None:
            continue
        dataset_index = int(study["datasetinfo"][study_set - 1].get("index") or study_set)
        normalized.update(
            {
                "member_index": member_index,
                "study_set": study_set,
                "dataset_index": dataset_index,
                "component": component,
                "subject": str(study["datasetinfo"][study_set - 1].get("subject") or ""),
                "coordformat": str((dataset.get("dipfit") or {}).get("coordformat") or ""),
            }
        )
        dipoles.append(normalized)
    if not dipoles:
        raise ValueError(f"STUDY cluster {cluster_index} has no localized dipoles in the selection")
    return {
        "cluster_index": cluster_index,
        "name": str(cluster.get("name") or f"Cluster {cluster_index}"),
        "dipoles": dipoles,
        "centroid": _centroid(dipoles),
        "coordformat": _coordinate_format(dipoles),
    }


def _dipole_model(model: dict[str, Any], cluster_index: int, component: int) -> dict[str, Any] | None:
    positions = np.asarray(model.get("posxyz", []), dtype=float)
    if positions.size == 0:
        return None
    positions = _xyz_matrix(positions, "posxyz", cluster_index, component)
    moments = _xyz_matrix(model.get("momxyz", []), "momxyz", cluster_index, component)
    if moments.shape != positions.shape:
        raise ValueError(f"STUDY cluster {cluster_index} component {component} posxyz and momxyz shapes differ")
    if positions.shape[0] == 2 and np.array_equal(positions[1], np.zeros(3)):
        positions = positions[:1]
        moments = moments[:1]
    rv = np.asarray(model.get("rv", []), dtype=float)
    if rv.size != 1 or not np.isfinite(rv.item()):
        raise ValueError(f"STUDY cluster {cluster_index} component {component} rv must be one finite value")
    return {"posxyz": positions.copy(), "momxyz": moments.copy(), "rv": float(rv.item())}


def _xyz_matrix(values: Any, name: str, cluster_index: int, component: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"STUDY cluster {cluster_index} component {component} {name} must be a finite n-by-3 array")
    return array


def _centroid(dipoles: list[dict[str, Any]]) -> dict[str, Any]:
    positions = np.stack([np.mean(dipole["posxyz"], axis=0) for dipole in dipoles])
    moments = np.stack([np.mean(dipole["momxyz"], axis=0) for dipole in dipoles])
    return {
        "posxyz": np.mean(positions, axis=0, keepdims=True),
        "momxyz": np.mean(moments, axis=0, keepdims=True),
        "rv": float(np.mean([dipole["rv"] for dipole in dipoles])),
    }


def _coordinate_format(dipoles: list[dict[str, Any]]) -> str:
    formats = {dipole["coordformat"].strip().lower() for dipole in dipoles if dipole["coordformat"].strip()}
    if len(formats) > 1:
        raise ValueError("Selected STUDY dipoles use incompatible coordinate formats")
    return next((dipole["coordformat"] for dipole in dipoles if dipole["coordformat"].strip()), "")


def _validate_coordinate_formats(selections: list[dict[str, Any]]) -> None:
    formats = {selection["coordformat"].strip().lower() for selection in selections if selection["coordformat"].strip()}
    if len(formats) > 1:
        raise ValueError("Selected STUDY clusters use incompatible coordinate formats")


def _plot_values(value: Any, count: int, default: Any) -> list[Any]:
    if value is None or _is_empty(value):
        if default == "tab10":
            return [plt.get_cmap(default)(index % 10) for index in range(count)]
        return [default] * count
    if isinstance(value, str) or np.asarray(value).ndim == 0:
        return [value] * count
    values = list(value)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"plot style must contain one value or {count} cluster values")
    return values


def _plot_selections(selections: list[dict[str, Any]], mode: str, colors: list[Any], sizes: list[Any]) -> list[Any]:
    if mode == "apart":
        return [
            _cluster_figure(selection, color, size)
            for selection, color, size in zip(selections, colors, sizes, strict=True)
        ]
    if mode == "together":
        figure = plt.figure(figsize=(6.2 * len(selections), 5.5))
        for index, (selection, color, size) in enumerate(zip(selections, colors, sizes, strict=True), start=1):
            axis = figure.add_subplot(1, len(selections), index, projection="3d")
            _plot_cluster(axis, selection, color, size, include_dipoles=True, include_centroid=True)
            axis.set_title(str(selection["name"]))
            _finish_axis(axis, [selection])
        figure.tight_layout()
        return [figure]
    figure = plt.figure(figsize=(7.2, 6.0))
    axis = figure.add_subplot(111, projection="3d")
    for selection, color, size in zip(selections, colors, sizes, strict=True):
        _plot_cluster(
            axis,
            selection,
            color,
            size,
            include_dipoles=mode != "centroid",
            include_centroid=mode != "comps",
        )
    axis.set_title("STUDY dipole centroids" if mode == "centroid" else "STUDY cluster dipoles")
    _finish_axis(axis, selections)
    figure.tight_layout()
    return [figure]


def _cluster_figure(selection: dict[str, Any], color: Any, size: Any) -> Any:
    figure = plt.figure(figsize=(7.2, 6.0))
    axis = figure.add_subplot(111, projection="3d")
    _plot_cluster(axis, selection, color, size, include_dipoles=True, include_centroid=True)
    axis.set_title(str(selection["name"]))
    _finish_axis(axis, [selection])
    figure.tight_layout()
    return figure


def _plot_cluster(
    axis: Any,
    selection: dict[str, Any],
    color: Any,
    size: Any,
    *,
    include_dipoles: bool,
    include_centroid: bool,
) -> None:
    if include_dipoles:
        for dipole in selection["dipoles"]:
            positions = dipole["posxyz"]
            moments = dipole["momxyz"]
            label = f"{selection['name']}: {dipole['subject']} IC{dipole['component']} (RV {dipole['rv'] * 100:.1f}%)"
            axis.scatter(*positions.T, s=float(size), color=color, label=label)
            axis.quiver(*positions.T, *moments.T, length=15.0, normalize=True, color=color)
    if include_centroid:
        centroid = selection["centroid"]
        label = f"{selection['name']} centroid (RV {centroid['rv'] * 100:.1f}%)"
        axis.scatter(
            *centroid["posxyz"].T,
            s=float(size) * 1.35,
            marker="D",
            color=color,
            edgecolor="black",
            label=label,
        )


def _finish_axis(axis: Any, selections: list[dict[str, Any]]) -> None:
    positions = [dipole["posxyz"] for selection in selections for dipole in selection["dipoles"]]
    values = np.concatenate(positions, axis=0)
    center = np.mean(values, axis=0)
    radius = max(float(np.max(np.linalg.norm(values - center, axis=1))), 10.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    formats = [selection["coordformat"] for selection in selections if selection["coordformat"]]
    if formats:
        axis.text2D(0.02, 0.98, formats[0], transform=axis.transAxes, va="top")
    axis.legend(fontsize=7)


def _is_empty(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple)) and not value


__all__ = ["std_dipplot"]

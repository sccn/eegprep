"""Plot STUDY component-cluster scalp maps."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.functions.sigprocfunc.topoplot import topoplot
from eegprep.functions.studyfunc._cluster_utils import cluster_list, sets_array
from eegprep.functions.studyfunc._study_utils import build_python_call, ensure_study
from eegprep.functions.studyfunc.std_readdata import std_readtopo


def std_topoplot(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    clusters: Any = "all",
    components: Any = None,
    mode: str = "together",
    figure: str | bool = "on",
    plotrad: float = 0.5,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Plot polarity-aligned cluster centroids or their component maps.

    Cluster and component indices are 1-based. For a child cluster, ``comps``
    selects member positions, matching EEGLAB's ``std_topoplot`` contract.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    clusters = options.pop("clusters", clusters)
    components = options.pop("components", options.pop("comps", components))
    mode = str(options.pop("mode", mode) or "together").lower()
    figure = options.pop("figure", figure)
    plotrad = float(options.pop("plotrad", plotrad))
    if "plotsubjects" in options:
        options.pop("plotsubjects")
        mode = "apart"
    if options:
        raise ValueError(f"Unknown std_topoplot option(s): {', '.join(sorted(options))}")
    mode = {"centroid": "together", "comps": "apart"}.get(mode, mode)
    if mode not in {"together", "apart"}:
        raise ValueError("mode must be 'together'/'centroid' or 'apart'/'comps'")

    study = ensure_study(STUDY)
    datasets = list(ALLEEG or [])
    if not datasets:
        raise ValueError("std_topoplot requires ALLEEG channel locations")
    selected = _cluster_indices(study, clusters)
    records = [_cluster_maps(study, datasets, index, components if len(selected) == 1 else None) for index in selected]
    _cache_cluster_maps(study, records)
    output_figure = (
        _plot_centroids(records, datasets, figure, plotrad)
        if mode == "together"
        else _plot_components(records, datasets, figure, plotrad)
    )
    command = build_python_call(
        ("STUDY", "FIGURE"),
        "std_topoplot",
        "STUDY",
        "ALLEEG",
        clusters=clusters,
        components=components,
        mode=mode,
        figure=figure,
        plotrad=plotrad,
    )
    return (study, output_figure, command) if return_com else (study, output_figure)


def _cluster_maps(
    study: dict[str, Any], datasets: list[dict[str, Any]], cluster_index: int, components: Any
) -> dict[str, Any]:
    _study, raw, _channel_axis = std_readtopo(study, datasets, clusters=cluster_index)
    maps = np.asarray(raw[0], dtype=float)
    cluster = cluster_list(study)[cluster_index - 1]
    if maps.ndim == 3:
        maps = maps.reshape(-1, maps.shape[-1])
        maps = maps[~np.isnan(maps).all(axis=1)]
    if maps.ndim != 2 or maps.shape[0] == 0:
        raise ValueError(f"cluster {cluster_index} contains no cached scalp maps")
    positions = _member_positions(components, maps.shape[0])
    maps = maps[positions]
    aligned, polarity = _align_polarity(maps)
    sets = sets_array(cluster.get("sets")).astype(int)
    comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
    labels = []
    for position in positions.tolist():
        if cluster_index == 1 or position >= comps.size or sets.size == 0:
            labels.append(f"Map {position + 1}")
            continue
        dataset_id = int(sets[0, position])
        subject = str((study.get("datasetinfo") or [{}])[dataset_id - 1].get("subject") or f"S{dataset_id}")
        labels.append(f"{subject}/IC{int(comps[position])}")
    return {
        "index": cluster_index,
        "name": str(cluster.get("name") or f"Cluster {cluster_index}"),
        "maps": aligned,
        "centroid": np.nanmean(aligned, axis=0),
        "polarity": polarity,
        "labels": labels,
    }


def _align_polarity(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.asarray(maps, dtype=float).copy()
    reference = aligned[0]
    polarity = np.ones(aligned.shape[0], dtype=int)
    for index in range(1, aligned.shape[0]):
        finite = np.isfinite(reference) & np.isfinite(aligned[index])
        if np.count_nonzero(finite) >= 2 and np.dot(reference[finite], aligned[index, finite]) < 0:
            aligned[index] *= -1
            polarity[index] = -1
    return aligned, polarity


def _cache_cluster_maps(study: dict[str, Any], records: list[dict[str, Any]]) -> None:
    clusters = deepcopy(cluster_list(study))
    for record in records:
        cluster = clusters[record["index"] - 1]
        cluster["topo"] = record["centroid"].tolist()
        cluster["topoall"] = record["maps"].tolist()
        cluster["topopol"] = record["polarity"].tolist()
    study["cluster"] = clusters
    study["saved"] = "no"


def _plot_centroids(records: list[dict[str, Any]], datasets: list[dict[str, Any]], figure: Any, plotrad: float) -> Any:
    rows, columns = _grid(len(records))
    fig = _figure(figure, rows, columns)
    axes = _axes(fig, rows, columns)
    chanlocs = list(datasets[0].get("chanlocs") or [])
    for axis, record in zip(axes.flat, records):
        _draw_topography(axis, record["centroid"], chanlocs, plotrad)
        axis.set_title(f"{record['name']} ({record['maps'].shape[0]} ICs)")
    _hide_unused(axes, len(records))
    fig.suptitle("Average scalp map for all clusters" if len(records) > 1 else records[0]["name"])
    setattr(fig, "eegprep_plot_metadata", {"mode": "centroid", "clusters": [record["index"] for record in records]})
    fig.tight_layout()
    return fig


def _plot_components(records: list[dict[str, Any]], datasets: list[dict[str, Any]], figure: Any, plotrad: float) -> Any:
    count = sum(record["maps"].shape[0] + 1 for record in records)
    rows, columns = _grid(count)
    fig = _figure(figure, rows, columns)
    axes = _axes(fig, rows, columns)
    chanlocs = list(datasets[0].get("chanlocs") or [])
    position = 0
    for record in records:
        _draw_topography(axes.flat[position], record["centroid"], chanlocs, plotrad)
        axes.flat[position].set_title(f"{record['name']} mean")
        position += 1
        for values, label in zip(record["maps"], record["labels"]):
            _draw_topography(axes.flat[position], values, chanlocs, plotrad)
            axes.flat[position].set_title(label)
            position += 1
    _hide_unused(axes, position)
    fig.suptitle("Cluster component scalp maps")
    setattr(
        fig,
        "eegprep_plot_metadata",
        {"mode": "components", "clusters": [record["index"] for record in records]},
    )
    fig.tight_layout()
    return fig


def _draw_topography(axis: Any, values: np.ndarray, chanlocs: list[dict[str, Any]], plotrad: float) -> None:
    if len(chanlocs) != values.size:
        raise ValueError("component scalp-map length does not match ALLEEG channel locations")
    topoplot(values, chanlocs, axes=axis, colorbar=False, plotrad=plotrad, intrad=plotrad)


def _cluster_indices(study: dict[str, Any], clusters: Any) -> list[int]:
    entries = cluster_list(study)
    if isinstance(clusters, str):
        if clusters.lower() != "all":
            raise ValueError("clusters must be numeric or 'all'")
        selected = [
            index
            for index, cluster in enumerate(entries[1:], start=2)
            if not str(cluster.get("name") or "").lower().startswith(("notclust", "parentcluster"))
        ]
        return selected or [1]
    values = numeric_vector(clusters, dtype=int)
    if values.size == 0:
        return _cluster_indices(study, "all")
    if np.any(values < 1) or np.any(values > len(entries)):
        raise ValueError(f"clusters must be 1-based and within 1..{len(entries)}")
    return values.astype(int).tolist()


def _member_positions(components: Any, count: int) -> np.ndarray:
    if components is None or (isinstance(components, str) and components.lower() == "all"):
        return np.arange(count, dtype=int)
    values = numeric_vector(components, dtype=int)
    if np.any(values < 1) or np.any(values > count):
        raise ValueError(f"comps must be 1-based cluster member positions within 1..{count}")
    return values - 1


def _grid(count: int) -> tuple[int, int]:
    columns = int(np.ceil(np.sqrt(count)))
    return int(np.ceil(count / columns)), columns


def _figure(value: Any, rows: int, columns: int) -> Any:
    if value is False or (isinstance(value, str) and value.lower() == "off"):
        fig = plt.gcf()
        fig.clear()
        return fig
    return plt.figure(figsize=(3.6 * columns, 3.4 * rows))


def _axes(fig: Any, rows: int, columns: int) -> np.ndarray:
    return np.asarray(
        [fig.add_subplot(rows, columns, index) for index in range(1, rows * columns + 1)], dtype=object
    ).reshape(rows, columns)


def _hide_unused(axes: np.ndarray, count: int) -> None:
    for axis in axes.flat[count:]:
        axis.set_visible(False)


__all__ = ["std_topoplot"]

"""Shared reading, grouping, statistics, and plotting for STUDY measures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.sigprocfunc.topoplot import topoplot
from eegprep.functions.studyfunc._cluster_utils import cluster_list
from eegprep.functions.studyfunc._study_measure_cells import (
    GroupedMeasure,
    group_channel_measures,
    group_component_measures,
)
from eegprep.functions.studyfunc._study_utils import MEASURE_DATA_FIELDS, build_python_call, range_mask
from eegprep.functions.studyfunc.std_readdata import std_readdata
from eegprep.functions.studyfunc.std_stat import StudyStatistics, std_stat


LINE_MEASURES = {"erp", "spec"}
PARAMETER_SECTIONS = {"erp": "erpparams", "spec": "specparams", "ersp": "erspparams", "itc": "erspparams"}
PARAMETER_OPTIONS = {
    "filter",
    "subtractsubjectmean",
    "timerange",
    "freqrange",
    "topotime",
    "topofreq",
    "averagechan",
    "detachplots",
    "ylim",
    "ersplim",
    "itclim",
    "maskdata",
    "averagemode",
    "subbaseline",
    "plotgroups",
    "plotconditions",
}
STATISTIC_OPTIONS = {
    "effect",
    "groupstats",
    "condstats",
    "singletrials",
    "statistics",
    "threshold",
    "alpha",
    "method",
    "mcorrect",
    "naccu",
}


def default_measure_target(study: dict[str, Any], field: str, channels: Any, clusters: Any, components: Any):
    """Choose channels-vs-parent-cluster default target by cached measure field."""
    if channels is not None or clusters is not None or components is not None:
        return channels, clusters
    if any(isinstance(group, dict) and field in group for group in study.get("changrp") or []):
        return "channels", clusters
    return channels, 1


def std_measureplot(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    datatype: str,
    *args: Any,
    channels: Any = None,
    clusters: Any = None,
    components: Any = None,
    design: int | None = None,
    noplot: str | bool = "off",
    plotmode: str = "normal",
    return_stats: bool = False,
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Read cached measures, arrange design cells, compute statistics, and plot."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    channels = options.pop("channels", channels)
    clusters = options.pop("clusters", clusters)
    components = options.pop("components", options.pop("comps", components))
    design = int(options.pop("design", design or STUDY.get("currentdesign") or 1))
    noplot = options.pop("noplot", noplot)
    plotmode = str(options.pop("plotmode", plotmode) or "normal").lower()
    return_stats = bool(options.pop("return_stats", return_stats))
    subject = options.pop("subject", None)
    plotsubjects = options.pop("plotsubjects", "off")
    mode = str(options.pop("mode", "") or "").lower()
    if mode in {"comps", "apart"}:
        plotsubjects = "on"
    elif mode not in {"", "centroid", "together"}:
        raise ValueError("mode must be 'centroid'/'together' or 'comps'/'apart'")
    options.pop("statmode", None)
    plotstderr = options.pop("plotstderr", "off")
    topoplotopt = options.pop("topoplotopt", None)
    caxis = options.pop("caxis", None)

    parameters, statistic_options = _plot_options(STUDY, datatype, options)
    timerange = parameters.get("timerange")
    freqrange = parameters.get("freqrange")
    plotconditions = str(parameters.get("plotconditions") or "apart").lower()
    plotgroups = str(parameters.get("plotgroups") or "apart").lower()
    if plotconditions not in {"apart", "together"} or plotgroups not in {"apart", "together"}:
        raise ValueError("plotconditions and plotgroups must be 'apart' or 'together'")

    channels, clusters = _default_target(STUDY, datatype, channels, clusters, components)
    study, grouped, x_axis, y_axis, target_titles = _read_grouped(
        STUDY,
        ALLEEG,
        datatype,
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
        subject=subject,
    )
    channel_names = _channel_names(study, channels) if channels is not None else None
    grouped = [_apply_group_ranges(item, datatype, x_axis, y_axis, timerange, freqrange) for item in grouped]
    x_axis = _axis_subset(x_axis, timerange if datatype != "spec" else freqrange)
    if datatype not in LINE_MEASURES:
        y_axis = _axis_subset(y_axis, freqrange)

    stats_config = _statistics_config(study, statistic_options, design=design)
    statistics = [std_stat(item.cells, stats_config, return_result=True) for item in grouped]
    topography = _topography_requested(parameters, datatype)
    if topography and channels is None:
        raise ValueError("topotime/topofreq scalp maps require channel measures")
    figure = None
    if not is_on(noplot) and plotmode != "none":
        figure = _plot_grouped_measures(
            grouped,
            datatype,
            x_axis,
            y_axis,
            statistics,
            target_titles,
            ALLEEG,
            topography=topography,
            channel_names=channel_names,
            topotime=parameters.get("topotime"),
            topofreq=parameters.get("topofreq"),
            plotconditions=plotconditions,
            plotgroups=plotgroups,
            plotsubjects=is_on(plotsubjects),
            plotstderr=is_on(plotstderr),
            caxis=caxis if caxis is not None else _measure_limits(parameters, datatype),
            topoplotopt=topoplotopt,
        )
    output_data = grouped[-1].output()
    output_statistics = statistics[-1]
    command = _history_command(
        datatype,
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
        noplot=noplot,
        plotmode=plotmode,
        subject=subject,
        plotsubjects=plotsubjects,
        return_stats=return_stats,
        **parameters,
        **statistic_options,
    )
    return _result(
        datatype,
        study,
        output_data,
        x_axis,
        y_axis,
        output_statistics,
        figure,
        command,
        return_stats=return_stats,
        return_com=return_com,
    )


def _plot_options(
    study: dict[str, Any], datatype: str, options: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = PARAMETER_SECTIONS[datatype]
    stored = (study.get("etc") or {}).get(section)
    parameters = deepcopy(stored) if isinstance(stored, dict) else {}
    statistics = {}
    unsupported = []
    for key, value in options.items():
        if key in PARAMETER_OPTIONS:
            parameters[key] = value
        elif key in STATISTIC_OPTIONS:
            statistics[key] = value
        else:
            unsupported.append(key)
    if unsupported:
        raise ValueError(f"Unknown std_{datatype}plot option(s): {', '.join(sorted(unsupported))}")
    return parameters, statistics


def _statistics_config(study: dict[str, Any], overrides: dict[str, Any], *, design: int) -> dict[str, Any]:
    stored = (study.get("etc") or {}).get("statistics")
    config = deepcopy(stored) if isinstance(stored, dict) else {}
    eeglab = config.get("eeglab")
    if not isinstance(eeglab, dict):
        eeglab = {}
    config["eeglab"] = eeglab
    for key, value in overrides.items():
        normalized = {"threshold": "alpha", "statistics": "method"}.get(key, key)
        if normalized in {"effect", "groupstats", "condstats", "singletrials", "mode"}:
            config[normalized] = value
        else:
            eeglab[normalized] = value
    variables = _design_variables(study, design)
    config["paired"] = [str(variable.get("pairing") or "off") for variable in variables[:2]]
    while len(config["paired"]) < 2:
        config["paired"].append("off")
    return config


def _read_grouped(
    study: dict[str, Any],
    alleeg: list[dict[str, Any]] | None,
    datatype: str,
    *,
    channels: Any,
    clusters: Any,
    components: Any,
    design: int,
    subject: Any,
) -> tuple[dict[str, Any], list[GroupedMeasure], np.ndarray, np.ndarray, list[str]]:
    if channels is not None:
        study, raw, x_axis, y_axis = std_readdata(study, alleeg, datatype=datatype, channels=channels)
        grouped = group_channel_measures(study, raw, datatype, design=design, subject=subject)
        names = _channel_names(study, channels)
        return study, [grouped], x_axis, y_axis, [", ".join(names)]

    cluster_indices = _cluster_indices(study, clusters)
    grouped_targets = []
    target_titles = []
    x_axis = np.asarray([])
    y_axis = np.asarray([])
    for cluster_index in cluster_indices:
        study, raw, current_x, current_y = std_readdata(
            study,
            alleeg,
            datatype=datatype,
            clusters=cluster_index,
            components=None,
        )
        x_axis = _shared_axis(x_axis, current_x, "measure")
        y_axis = _shared_axis(y_axis, current_y, "frequency")
        grouped_targets.append(
            group_component_measures(
                study,
                raw[0],
                datatype,
                cluster_index=cluster_index,
                components=components,
                design=design,
                subject=subject,
            )
        )
        target_titles.append(str(cluster_list(study)[cluster_index - 1].get("name") or f"Cluster {cluster_index}"))
    return study, grouped_targets, x_axis, y_axis, target_titles


def _shared_axis(existing: np.ndarray, current: np.ndarray, name: str) -> np.ndarray:
    current = np.asarray(current, dtype=float)
    if existing.size == 0:
        return current
    if existing.shape != current.shape or not np.allclose(existing, current):
        raise ValueError(f"selected targets do not share a common {name} axis")
    return existing


def _apply_group_ranges(
    grouped: GroupedMeasure,
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    timerange: Any,
    freqrange: Any,
) -> GroupedMeasure:
    x_mask = _axis_mask(x_axis, timerange if datatype != "spec" else freqrange)
    y_mask = _axis_mask(y_axis, freqrange) if datatype not in LINE_MEASURES else None
    cells = []
    for row in grouped.cells:
        selected_row = []
        for values in row:
            array = np.asarray(values)
            if datatype in LINE_MEASURES:
                selected_row.append(array if x_mask is None else array[x_mask, ...])
            else:
                if y_mask is not None:
                    array = array[y_mask, ...]
                if x_mask is not None:
                    array = array[:, x_mask, ...]
                selected_row.append(array)
        cells.append(selected_row)
    return GroupedMeasure(cells, grouped.conditions, grouped.groups, grouped.cases)


def _plot_grouped_measures(
    targets: list[GroupedMeasure],
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    statistics: list[StudyStatistics],
    target_titles: list[str],
    alleeg: list[dict[str, Any]] | None,
    **options: Any,
) -> Any:
    if options.pop("topography"):
        return _plot_channel_topographies(targets[0], datatype, x_axis, y_axis, statistics[0], alleeg, **options)
    if len(targets) > 1:
        return _plot_multiple_targets(targets, datatype, x_axis, y_axis, statistics, target_titles, **options)
    grouped = targets[0]
    if datatype in LINE_MEASURES:
        return _plot_grouped_lines(grouped, datatype, x_axis, statistics[0], title=target_titles[0], **options)
    return _plot_grouped_images(grouped, datatype, x_axis, y_axis, statistics[0], title=target_titles[0], **options)


def _plot_grouped_lines(
    grouped: GroupedMeasure,
    datatype: str,
    x_axis: np.ndarray,
    statistics: StudyStatistics,
    *,
    title: str,
    plotconditions: str,
    plotgroups: str,
    plotsubjects: bool,
    plotstderr: bool,
    caxis: Any,
    **_options: Any,
) -> Any:
    rows = 1 if plotconditions == "together" else len(grouped.conditions)
    columns = 1 if plotgroups == "together" else len(grouped.groups)
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5 * columns, 3.5 * rows))
    for condition_index, condition in enumerate(grouped.conditions):
        for group_index, group in enumerate(grouped.groups):
            axis = axes[0 if rows == 1 else condition_index, 0 if columns == 1 else group_index]
            values = grouped.cells[condition_index][group_index]
            _draw_line_cell(
                axis,
                x_axis,
                values,
                f"{condition} / {group}",
                plotsubjects=plotsubjects,
                plotstderr=plotstderr,
            )
            _draw_line_significance(axis, x_axis, statistics, condition_index, group_index)
    for row in axes:
        for axis in row:
            axis.set_xlabel("Time (ms)" if datatype == "erp" else "Frequency (Hz)")
            axis.set_ylabel("uV" if datatype == "erp" else "Power 10*log10(uV^2/Hz)")
            axis.grid(True, alpha=0.25)
            if caxis is not None and np.asarray(caxis).size == 2:
                axis.set_ylim(np.asarray(caxis, dtype=float).ravel())
            if axis.lines:
                axis.legend(fontsize=8)
    fig.suptitle(f"STUDY {datatype.upper()} — {title}")
    _attach_metadata(fig, grouped, statistics)
    fig.tight_layout()
    return fig


def _draw_line_cell(
    axis: Any,
    x_axis: np.ndarray,
    values: np.ndarray,
    label: str,
    *,
    plotsubjects: bool,
    plotstderr: bool,
) -> None:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = np.nanmean(array, axis=1)
    if array.ndim != 2:
        raise ValueError("grouped line cells must be samples x cases, with an optional channel axis")
    if array.shape[-1] == 0:
        axis.set_title(f"{label} (no observations)")
        return
    if plotsubjects:
        for case_index in range(array.shape[-1]):
            axis.plot(x_axis, array[:, case_index], color="0.7", linewidth=0.8, alpha=0.8)
    mean = np.nanmean(array, axis=-1)
    axis.plot(x_axis, mean, linewidth=2.0, label=label)
    if plotstderr and array.shape[-1] > 1:
        sem = np.nanstd(array, axis=-1, ddof=1) / np.sqrt(array.shape[-1])
        axis.fill_between(x_axis, mean - sem, mean + sem, alpha=0.2)


def _draw_line_significance(
    axis: Any,
    x_axis: np.ndarray,
    statistics: StudyStatistics,
    condition_index: int,
    group_index: int,
) -> None:
    masks = []
    if group_index < len(statistics.condmask):
        masks.append(statistics.condmask[group_index])
    if condition_index < len(statistics.groupmask):
        masks.append(statistics.groupmask[condition_index])
    if not masks:
        return
    mask = np.logical_or.reduce([np.asarray(value, dtype=bool) for value in masks])
    while mask.ndim > 1:
        mask = np.any(mask, axis=-1)
    if mask.size == x_axis.size and np.any(mask):
        axis.fill_between(x_axis, 0, 1, where=mask, color="gold", alpha=0.14, transform=axis.get_xaxis_transform())


def _plot_grouped_images(
    grouped: GroupedMeasure,
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    statistics: StudyStatistics,
    *,
    title: str,
    caxis: Any,
    plotsubjects: bool,
    **_options: Any,
) -> Any:
    panels = _image_panels(grouped, datatype, plotsubjects=plotsubjects)
    columns = int(np.ceil(np.sqrt(len(panels))))
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5 * columns, 3.8 * rows))
    limits = np.asarray(caxis, dtype=float).ravel() if caxis is not None else np.asarray([])
    for axis, panel in zip(axes.flat, panels):
        condition_index, group_index, label, image = panel
        kwargs = {"vmin": limits[0], "vmax": limits[1]} if limits.size == 2 else {}
        mesh = axis.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
            **kwargs,
        )
        mask = _image_mask(statistics, condition_index, group_index)
        if mask is not None and mask.shape == image.shape and np.any(mask):
            axis.contour(x_axis, y_axis, mask.astype(float), levels=[0.5], colors="white", linewidths=1.0)
        axis.set_title(label)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Frequency (Hz)")
        fig.colorbar(mesh, ax=axis)
    for axis in axes.flat[len(panels) :]:
        axis.set_visible(False)
    fig.suptitle(f"STUDY {datatype.upper()} — {title}")
    _attach_metadata(fig, grouped, statistics)
    fig.tight_layout()
    return fig


def _image_panels(
    grouped: GroupedMeasure, datatype: str, *, plotsubjects: bool
) -> list[tuple[int, int, str, np.ndarray]]:
    panels = []
    for condition_index, condition in enumerate(grouped.conditions):
        for group_index, group in enumerate(grouped.groups):
            cell = grouped.cells[condition_index][group_index]
            if plotsubjects and cell.shape[-1]:
                for case_index, case in enumerate(grouped.cases[condition_index][group_index]):
                    panels.append(
                        (condition_index, group_index, f"{condition} / {group} — {case}", _case_image(cell, case_index))
                    )
            else:
                panels.append((condition_index, group_index, f"{condition} / {group}", _mean_image(cell, datatype)))
    return panels


def _case_image(values: np.ndarray, case_index: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)[..., case_index]
    return np.nanmean(array, axis=-1) if array.ndim == 3 else array


def _mean_image(values: np.ndarray, datatype: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 4:
        array = np.sqrt(np.nanmean(array**2, axis=2)) if datatype == "ersp" else np.nanmean(array, axis=2)
    if array.ndim != 3:
        raise ValueError("grouped time-frequency cells must be frequency x time x cases")
    if array.shape[-1] == 0:
        return np.full(array.shape[:2], np.nan)
    return np.nanmean(array, axis=-1)


def _image_mask(statistics: StudyStatistics, condition_index: int, group_index: int) -> np.ndarray | None:
    masks = []
    if group_index < len(statistics.condmask):
        masks.append(statistics.condmask[group_index])
    if condition_index < len(statistics.groupmask):
        masks.append(statistics.groupmask[condition_index])
    if not masks:
        return None
    mask = np.logical_or.reduce([np.asarray(value, dtype=bool) for value in masks])
    while mask.ndim > 2:
        mask = np.any(mask, axis=-1)
    return mask


def _plot_multiple_targets(
    targets: list[GroupedMeasure],
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    statistics: list[StudyStatistics],
    titles: list[str],
    **options: Any,
) -> Any:
    columns = int(np.ceil(np.sqrt(len(targets))))
    rows = int(np.ceil(len(targets) / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(5 * columns, 3.8 * rows))
    for index, (grouped, title) in enumerate(zip(targets, titles)):
        axis = axes.flat[index]
        cell_means = []
        for row in grouped.cells:
            for cell in row:
                array = np.asarray(cell, dtype=float)
                if datatype in LINE_MEASURES:
                    while array.ndim > 2:
                        array = np.nanmean(array, axis=1)
                    cell_means.append(np.nanmean(array, axis=-1))
                else:
                    cell_means.append(_mean_image(array, datatype))
        mean = np.nanmean(np.stack(cell_means), axis=0)
        if datatype in LINE_MEASURES:
            axis.plot(x_axis, mean)
            axis.set_xlabel("Time (ms)" if datatype == "erp" else "Frequency (Hz)")
        else:
            axis.imshow(
                mean,
                aspect="auto",
                origin="lower",
                extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
            )
        axis.set_title(title)
    for axis in axes.flat[len(targets) :]:
        axis.set_visible(False)
    fig.suptitle(f"STUDY {datatype.upper()} clusters")
    setattr(fig, "eegprep_plot_metadata", {"targets": titles, "statistics": statistics})
    fig.tight_layout()
    return fig


def _plot_channel_topographies(
    grouped: GroupedMeasure,
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    statistics: StudyStatistics,
    alleeg: list[dict[str, Any]] | None,
    *,
    channel_names: list[str] | None,
    topotime: Any,
    topofreq: Any,
    caxis: Any,
    topoplotopt: Any,
    **_options: Any,
) -> Any:
    datasets = list(alleeg or [])
    if not datasets:
        raise ValueError("channel topographies require ALLEEG channel locations")
    channel_count = _channel_count(grouped.cells[0][0], datatype)
    if channel_names is None or len(channel_names) != channel_count:
        raise ValueError("channel topographies require names for every selected channel")
    available = {
        str(location.get("labels") or "").lower(): location for location in (datasets[0].get("chanlocs") or [])
    }
    missing = [name for name in channel_names if name.lower() not in available]
    if missing:
        raise ValueError(f"channel locations are missing for: {', '.join(missing)}")
    chanlocs = [available[name.lower()] for name in channel_names]
    if len(chanlocs) != channel_count or channel_count < 2:
        raise ValueError("channel topographies require matching channel locations")
    rows = len(grouped.conditions)
    columns = len(grouped.groups)
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4 * columns, 3.8 * rows))
    limits = np.asarray(caxis, dtype=float).ravel() if caxis is not None else np.asarray([])
    topo_kwargs = _topoplot_options(topoplotopt)
    if limits.size == 2:
        topo_kwargs["maplimits"] = limits
    for condition_index, condition in enumerate(grouped.conditions):
        for group_index, group in enumerate(grouped.groups):
            values = _topography_values(
                grouped.cells[condition_index][group_index], datatype, x_axis, y_axis, topotime, topofreq
            )
            axis = axes[condition_index, group_index]
            if np.isfinite(values).any():
                topoplot(values, chanlocs, axes=axis, colorbar=False, **topo_kwargs)
            else:
                axis.axis("off")
            axis.set_title(f"{condition} / {group}")
    fig.suptitle(f"STUDY {datatype.upper()} scalp map")
    _attach_metadata(fig, grouped, statistics)
    fig.tight_layout()
    return fig


def _channel_count(values: np.ndarray, datatype: str) -> int:
    array = np.asarray(values)
    expected = 3 if datatype in LINE_MEASURES else 4
    if array.ndim != expected:
        raise ValueError("scalp-map plotting requires two or more selected channels")
    return int(array.shape[-2])


def _topography_values(
    values: np.ndarray,
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    topotime: Any,
    topofreq: Any,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if datatype in LINE_MEASURES:
        bounds = topotime if datatype == "erp" else topofreq
        mask = _nearest_or_range_mask(x_axis, bounds)
        return np.nanmean(array[mask, ...], axis=(0, -1))
    time_mask = _nearest_or_range_mask(x_axis, topotime)
    frequency_mask = _nearest_or_range_mask(y_axis, topofreq)
    selected = array[frequency_mask, ...][:, time_mask, ...]
    return np.nanmean(selected, axis=(0, 1, -1))


def _nearest_or_range_mask(axis: np.ndarray, bounds: Any) -> np.ndarray:
    values = np.asarray(bounds if bounds is not None else [], dtype=float).ravel()
    if values.size == 0:
        return np.ones(axis.size, dtype=bool)
    if values.size == 1:
        mask = np.zeros(axis.size, dtype=bool)
        mask[int(np.argmin(np.abs(axis - values[0])))] = True
        return mask
    return (axis >= np.min(values)) & (axis <= np.max(values))


def _topoplot_options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, (list, tuple)) or len(value) % 2:
        raise ValueError("topoplotopt must be a dictionary or key/value sequence")
    return {str(value[index]).lower(): value[index + 1] for index in range(0, len(value), 2)}


def _attach_metadata(figure: Any, grouped: GroupedMeasure, statistics: StudyStatistics) -> None:
    setattr(
        figure,
        "eegprep_plot_metadata",
        {
            "conditions": grouped.conditions,
            "groups": grouped.groups,
            "cases": grouped.cases,
            "statistics": statistics,
        },
    )


def _topography_requested(parameters: dict[str, Any], datatype: str) -> bool:
    if datatype == "erp":
        return _has_values(parameters.get("topotime"))
    if datatype == "spec":
        return _has_values(parameters.get("topofreq"))
    return _has_values(parameters.get("topotime")) and _has_values(parameters.get("topofreq"))


def _measure_limits(parameters: dict[str, Any], datatype: str) -> Any:
    return parameters.get({"erp": "ylim", "spec": "ylim", "ersp": "ersplim", "itc": "itclim"}[datatype])


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    array = np.asarray(value).reshape(-1)
    return bool(array.size and not np.isnan(np.asarray(array, dtype=float)[0]))


def _channel_names(study: dict[str, Any], channels: Any) -> list[str]:
    groups = [group for group in study.get("changrp") or [] if isinstance(group, dict)]
    if isinstance(channels, str) and channels == "channels":
        return [str(group.get("name") or "channel") for group in groups]
    if isinstance(channels, str):
        return [channels]
    if isinstance(channels, (list, tuple)) and channels and isinstance(channels[0], str):
        return [str(value) for value in channels]
    indices = np.asarray(channels if channels is not None else [], dtype=int).ravel()
    if indices.size == 0:
        return [str(group.get("name") or "channel") for group in groups]
    return [str(groups[int(index) - 1].get("name") or index) for index in indices]


def _cluster_indices(study: dict[str, Any], clusters: Any) -> list[int]:
    entries = cluster_list(study)
    if isinstance(clusters, str) and clusters.lower() == "all":
        selected = [index for index in range(2, len(entries) + 1) if not _excluded_cluster(entries[index - 1])]
        return selected or [1]
    values = np.asarray(clusters if clusters is not None else [1], dtype=int).ravel()
    if not values.size or np.any(values < 1) or np.any(values > len(entries)):
        raise ValueError(f"clusters must be 1-based and within 1..{len(entries)}")
    return values.astype(int).tolist()


def _excluded_cluster(cluster: dict[str, Any]) -> bool:
    name = str(cluster.get("name") or "").lower()
    return name.startswith("notclust") or name.startswith("parentcluster")


def _design_variables(study: dict[str, Any], design: int) -> list[dict[str, Any]]:
    designs = study.get("design") or []
    if not designs:
        return []
    if design < 1 or design > len(designs):
        raise ValueError(f"design must be 1-based and within 1..{len(designs)}")
    return [value for value in designs[design - 1].get("variable") or [] if isinstance(value, dict)]


def _default_target(
    study: dict[str, Any], datatype: str, channels: Any, clusters: Any, components: Any
) -> tuple[Any, Any]:
    return default_measure_target(study, MEASURE_DATA_FIELDS[datatype], channels, clusters, components)


def _axis_subset(axis: np.ndarray, bounds: Any) -> np.ndarray:
    mask = _axis_mask(axis, bounds)
    return axis if mask is None else axis[mask]


def _axis_mask(axis: np.ndarray, bounds: Any) -> np.ndarray | None:
    mask = range_mask(
        axis,
        bounds,
        name="range options",
        empty_message="range option does not include any measure samples",
    )
    if mask.size == 0 or np.all(mask):
        return None
    return mask


def plot_measure_data(
    data: list[np.ndarray],
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    title: str | None = None,
    line_labels: list[str] | None = None,
) -> Any:
    """Plot raw cached arrays (the compact contract used by ``pop_chanplot``)."""
    if datatype in LINE_MEASURES:
        return _plot_lines(data, datatype, x_axis, title=title, line_labels=line_labels)
    return _plot_image(data, datatype, x_axis, y_axis, title=title)


def _plot_lines(
    data: list[np.ndarray],
    datatype: str,
    x_axis: np.ndarray,
    *,
    title: str | None,
    line_labels: list[str] | None,
) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in _line_series(data, line_labels):
        ax.plot(x_axis, values, label=label)
    ax.set_xlabel("Time (ms)" if datatype == "erp" else "Frequency (Hz)")
    ax.set_ylabel("uV" if datatype == "erp" else "Power 10*log10(uV^2/Hz)")
    ax.set_title(title or f"STUDY {datatype.upper()}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _line_series(data: list[np.ndarray], line_labels: list[str] | None) -> list[tuple[str, np.ndarray]]:
    series = []
    labels = iter(line_labels or [])
    for group_index, values in enumerate(data, start=1):
        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            series.append((next(labels, f"Group {group_index}"), np.nanmean(array, axis=0)))
        elif array.ndim == 3:
            for component_index, component_values in enumerate(np.nanmean(array, axis=0), start=1):
                series.append((next(labels, f"IC {component_index}"), component_values))
        else:
            raise ValueError("line measure data must be 2-D or 3-D")
    return series


def _plot_image(
    data: list[np.ndarray], datatype: str, x_axis: np.ndarray, y_axis: np.ndarray, *, title: str | None
) -> Any:
    images = []
    for values in data:
        array = np.asarray(values, dtype=float)
        if array.ndim == 3:
            images.append(np.nanmean(array, axis=0))
        elif array.ndim == 4:
            images.append(np.nanmean(array, axis=(0, 1)))
        else:
            raise ValueError("time-frequency measure data must be 3-D or 4-D")
    image = np.nanmean(np.stack(images, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    mesh = ax.imshow(
        image,
        aspect="auto",
        origin="lower",
        extent=[float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1])],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or f"STUDY {datatype.upper()}")
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    return fig


def _history_command(datatype: str, **kwargs: Any) -> str:
    if datatype in LINE_MEASURES:
        targets = ("STUDY", f"{datatype.upper()}DATA", _line_axis_name(datatype), "FIGURE")
    else:
        targets = ("STUDY", f"{datatype.upper()}DATA", f"{datatype.upper()}TIMES", f"{datatype.upper()}FREQS", "FIGURE")
    if kwargs.pop("return_stats", False):
        targets = (*targets[:-1], "PGROUP", "PCOND", "PINTER", targets[-1])
        kwargs["return_stats"] = True
    return build_python_call(targets, f"std_{datatype}plot", "STUDY", "ALLEEG", **kwargs)


def _line_axis_name(datatype: str) -> str:
    return "ERPTIMES" if datatype == "erp" else "SPECFREQS"


def _result(
    datatype: str,
    study: dict[str, Any],
    data: Any,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    statistics: StudyStatistics,
    figure: Any,
    command: str,
    *,
    return_stats: bool,
    return_com: bool,
) -> tuple[Any, ...]:
    result = (study, data, x_axis) if datatype in LINE_MEASURES else (study, data, x_axis, y_axis)
    if return_stats:
        pcond, pgroup, pinter = statistics.output()
        result = (*result, pgroup, pcond, pinter)
    result = (*result, figure)
    return (*result, command) if return_com else result


__all__ = ["default_measure_target", "plot_measure_data", "std_measureplot"]

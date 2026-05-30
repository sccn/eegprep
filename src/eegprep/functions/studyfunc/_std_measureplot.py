"""Shared plotting helpers for STUDY cached measures."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import build_python_call
from eegprep.functions.studyfunc.std_readdata import std_readdata


LINE_MEASURES = {"erp", "spec"}


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
    return_com: bool = False,
    **kwargs: Any,
) -> tuple[Any, ...]:
    """Read and optionally plot cached STUDY measure data."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    channels = options.pop("channels", channels)
    clusters = options.pop("clusters", clusters)
    components = options.pop("components", options.pop("comps", components))
    design = options.pop("design", design)
    noplot = options.pop("noplot", noplot)
    plotmode = str(options.pop("plotmode", plotmode) or "normal").lower()
    timerange = options.pop("timerange", None)
    freqrange = options.pop("freqrange", None)
    ignored = {"subject", "plotsubjects", "topoplotopt", "plotconditions", "mode"}
    unsupported = sorted(key for key in options if key not in ignored)
    if unsupported:
        raise ValueError(f"Unknown std_{datatype}plot option(s): {', '.join(unsupported)}")

    channels, clusters = _default_target(STUDY, datatype, channels, clusters, components)
    study, data, x_axis, y_axis = std_readdata(
        STUDY,
        ALLEEG,
        datatype=datatype,
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
    )
    data, x_axis, y_axis = _apply_ranges(data, datatype, x_axis, y_axis, timerange, freqrange)
    figure = None if is_on(noplot) or plotmode == "none" else _plot_measure(data, datatype, x_axis, y_axis)
    command = _history_command(
        datatype,
        channels=channels,
        clusters=clusters,
        components=components,
        design=design,
        noplot=noplot,
        plotmode=plotmode,
        timerange=timerange,
        freqrange=freqrange,
    )
    return _result(datatype, study, data, x_axis, y_axis, figure, command, return_com=return_com)


def _default_target(
    study: dict[str, Any], datatype: str, channels: Any, clusters: Any, components: Any
) -> tuple[Any, Any]:
    if channels is not None or clusters is not None or components is not None:
        return channels, clusters
    field = {"erp": "erpdata", "spec": "specdata", "ersp": "erspdata", "itc": "itcdata"}[datatype]
    if any(isinstance(group, dict) and field in group for group in study.get("changrp") or []):
        return "channels", clusters
    return channels, 1


def _apply_ranges(
    data: list[np.ndarray],
    datatype: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    timerange: Any,
    freqrange: Any,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    if datatype == "erp":
        return _select_last_axis(data, x_axis, timerange), _axis_subset(x_axis, timerange), y_axis
    if datatype == "spec":
        return _select_last_axis(data, x_axis, freqrange), _axis_subset(x_axis, freqrange), y_axis
    data = _select_last_axis(data, x_axis, timerange)
    x_axis = _axis_subset(x_axis, timerange)
    data = _select_frequency_axis(data, y_axis, freqrange)
    y_axis = _axis_subset(y_axis, freqrange)
    return data, x_axis, y_axis


def _select_last_axis(data: list[np.ndarray], axis: np.ndarray, bounds: Any) -> list[np.ndarray]:
    mask = _axis_mask(axis, bounds)
    if mask is None:
        return data
    return [np.asarray(values)[..., mask] for values in data]


def _select_frequency_axis(data: list[np.ndarray], axis: np.ndarray, bounds: Any) -> list[np.ndarray]:
    mask = _axis_mask(axis, bounds)
    if mask is None:
        return data
    selected = []
    for values in data:
        array = np.asarray(values)
        selected.append(array[..., mask, :])
    return selected


def _axis_subset(axis: np.ndarray, bounds: Any) -> np.ndarray:
    mask = _axis_mask(axis, bounds)
    return axis if mask is None else axis[mask]


def _axis_mask(axis: np.ndarray, bounds: Any) -> np.ndarray | None:
    if bounds is None or (isinstance(bounds, str) and bounds == ""):
        return None
    values = np.asarray(bounds, dtype=float).ravel()
    if values.size == 0:
        return None
    if values.size != 2:
        raise ValueError("range options must contain [min max]")
    mask = (axis >= values[0]) & (axis <= values[1])
    if not np.any(mask):
        raise ValueError("range option does not include any measure samples")
    return mask


def _plot_measure(data: list[np.ndarray], datatype: str, x_axis: np.ndarray, y_axis: np.ndarray) -> Any:
    if datatype in LINE_MEASURES:
        return _plot_lines(data, datatype, x_axis)
    return _plot_image(data, datatype, x_axis, y_axis)


def _plot_lines(data: list[np.ndarray], datatype: str, x_axis: np.ndarray) -> Any:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, values in _line_series(data):
        ax.plot(x_axis, values, label=label)
    ax.set_xlabel("Time (ms)" if datatype == "erp" else "Frequency (Hz)")
    ax.set_ylabel("uV" if datatype == "erp" else "Power 10*log10(uV^2/Hz)")
    ax.set_title(f"STUDY {datatype.upper()}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _line_series(data: list[np.ndarray]) -> list[tuple[str, np.ndarray]]:
    series = []
    for group_index, values in enumerate(data, start=1):
        array = np.asarray(values, dtype=float)
        if array.ndim == 2:
            series.append((f"Group {group_index}", np.nanmean(array, axis=0)))
        elif array.ndim == 3:
            for component_index, component_values in enumerate(np.nanmean(array, axis=0), start=1):
                series.append((f"IC {component_index}", component_values))
        else:
            raise ValueError("line measure data must be 2-D or 3-D")
    return series


def _plot_image(data: list[np.ndarray], datatype: str, x_axis: np.ndarray, y_axis: np.ndarray) -> Any:
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
    ax.set_title(f"STUDY {datatype.upper()}")
    fig.colorbar(mesh, ax=ax)
    fig.tight_layout()
    return fig


def _history_command(datatype: str, **kwargs: Any) -> str:
    return build_python_call(("STUDY",), f"std_{datatype}plot", "STUDY", "ALLEEG", **kwargs)


def _result(
    datatype: str,
    study: dict[str, Any],
    data: list[np.ndarray],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    figure: Any,
    command: str,
    *,
    return_com: bool,
) -> tuple[Any, ...]:
    if datatype in LINE_MEASURES:
        result = (study, data, x_axis, figure)
    else:
        result = (study, data, x_axis, y_axis, figure)
    return (*result, command) if return_com else result

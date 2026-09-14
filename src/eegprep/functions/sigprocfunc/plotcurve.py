"""Plot one or more curves with EEGLAB-style significance highlighting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from eegprep.functions.sigprocfunc.topoplot import topoplot


DEFAULT_CURVE_COLORS = ("r", "g", "b", "c", "m")


def plotcurve(times: Any, data: Any, *args: Any, target: Axes | None = None, **kwargs: Any) -> Axes:
    """Plot curves and highlight samples outside significance limits.

    ``data`` is normalized to ``(curves, times)``. Options may be passed as
    EEGLAB-style key/value positional pairs or Python keywords. Supported
    options include ``maskarray``, ``val2mask``, ``highlightmode``,
    ``plotmean``, ``plotindiv``, ``plotstderr``, labels, limits, legends,
    vertical markers, colors, and scalp-map insets.

    Args:
        times: One-dimensional time coordinates.
        data: Curve values with one dimension matching ``times``.
        *args: Optional EEGLAB-style key/value pairs.
        target: Existing axes. The current axes is used by default.
        **kwargs: Plot options named as in EEGLAB ``plotcurve``.

    Returns:
        The primary curve axes.
    """
    options = _parse_options(args, kwargs)
    time_values = np.asarray(times, dtype=float).ravel()
    if time_values.size == 0 or not np.isfinite(time_values).all():
        raise ValueError("plotcurve times must contain finite values")
    curves = _as_curves(data, time_values.size)
    ax = target or plt.gca()

    plot_stderr = options["plotstderr"]
    if plot_stderr is not None and np.asarray(plot_stderr).size:
        center = np.nanmean(curves, axis=0)
        error = np.broadcast_to(np.asarray(plot_stderr, dtype=float).squeeze(), center.shape)
        ax.fill_between(
            time_values,
            center - error,
            center + error,
            color=_style_parts(options["colors"][0])[0],
            alpha=float(options["transparent"]),
            edgecolor="none",
        )

    plotted = _curves_to_plot(curves, options["plotmean"], options["plotindiv"])
    lines = []
    for index, values in enumerate(plotted):
        is_mean = options["plotmean"] and options["plotindiv"] and index == plotted.shape[0] - 1
        color, linestyle = ("k", "-") if is_mean else _style_parts(options["colors"][index % len(options["colors"])])
        (line,) = ax.plot(
            time_values,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2 if is_mean else 1.5,
        )
        lines.append(line)

    y_limits = _y_limits(plotted, options["ylim"], plot_stderr)
    ax.set_ylim(y_limits)
    ax.set_xlim(float(time_values[0]), float(time_values[-1]))

    mask = options["maskarray"]
    if mask is not None and np.asarray(mask).size:
        comparison = curves if options["val2mask"] is None else _comparison_values(options["val2mask"], curves.shape)
        regions = _significant_regions(mask, comparison, time_values.size)
        _highlight_regions(ax, time_values, regions, options["highlightmode"], options["xlabel"])

    for value in np.asarray(options["marktimes"], dtype=float).ravel():
        if np.isfinite(value):
            ax.axvline(value, color="k", linestyle="--", linewidth=float(options["linewidth"]))
    for value in np.asarray(options["vert"], dtype=float).ravel():
        if np.isfinite(value):
            ax.axvline(value, color="m", linewidth=1)

    if options["plottopo"] is not None and np.asarray(options["plottopo"]).size:
        _plot_topographies(ax, options["plottopo"], options["chanlocs"], options["plottopotitle"])
    _finish_axes(ax, lines, options)
    return ax


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2:
        raise ValueError("plotcurve optional arguments must be key/value pairs")
    supplied = {str(args[index]).lower(): args[index + 1] for index in range(0, len(args), 2)}
    supplied.update({str(key).lower(): value for key, value in kwargs.items()})
    defaults: dict[str, Any] = {
        "maskarray": None,
        "val2mask": None,
        "highlightmode": "background",
        "plotmean": False,
        "plotindiv": True,
        "traceinfo": "off",
        "logpval": False,
        "title": "",
        "xlabel": "",
        "plotmode": "single",
        "plotstderr": None,
        "ylabel": "",
        "legend": (),
        "transparent": 0.5,
        "colors": DEFAULT_CURVE_COLORS,
        "plottopotitle": (),
        "chanlocs": None,
        "ylim": None,
        "vert": (),
        "plottopo": None,
        "linewidth": 2,
        "marktimes": (),
    }
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported plotcurve option: {sorted(unknown)[0]}")
    defaults.update(supplied)
    defaults["plotmean"] = _is_on(defaults["plotmean"])
    defaults["plotindiv"] = _is_on(defaults["plotindiv"])
    defaults["logpval"] = _is_on(defaults["logpval"])
    if not defaults["plotindiv"]:
        defaults["plotmean"] = True
    defaults["highlightmode"] = str(defaults["highlightmode"]).lower()
    if defaults["highlightmode"] not in {"background", "bottom"}:
        raise ValueError("plotcurve highlightmode must be 'background' or 'bottom'")
    defaults["plotmode"] = str(defaults["plotmode"]).lower()
    if defaults["plotmode"] not in {"single", "topo"}:
        raise ValueError("plotcurve plotmode must be 'single' or 'topo'")
    colors = defaults["colors"]
    defaults["colors"] = tuple(colors) if not isinstance(colors, str) else (colors,)
    if not defaults["colors"]:
        defaults["colors"] = DEFAULT_CURVE_COLORS
    return defaults


def _as_curves(data: Any, time_count: int) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        raise ValueError("plotcurve data must not be empty")
    values = np.squeeze(values)
    if values.ndim == 1:
        if values.size != time_count:
            raise ValueError("Size of time input and array input does not match")
        return values.reshape(1, time_count)
    if values.shape[-1] == time_count:
        return values.reshape(-1, time_count)
    if values.shape[0] == time_count:
        return np.moveaxis(values, 0, -1).reshape(-1, time_count)
    if values.size % time_count:
        raise ValueError("Size of time input and array input does not match")
    return values.reshape(-1, time_count)


def _curves_to_plot(curves: np.ndarray, plot_mean: bool, plot_individual: bool) -> np.ndarray:
    mean = np.nanmean(curves, axis=0, keepdims=True)
    if not plot_individual:
        return mean
    if plot_mean:
        return np.vstack([curves, mean])
    return curves


def _comparison_values(value: Any, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(shape, float(values))
    return np.broadcast_to(values, shape)


def _significant_regions(mask: Any, values: np.ndarray, time_count: int) -> np.ndarray:
    limits = np.asarray(mask)
    if limits.ndim == 1 and limits.size == time_count and np.all(np.isin(np.unique(limits), [0, 1])):
        return limits.astype(bool)
    numeric = np.asarray(mask, dtype=float)
    if numeric.ndim == 1 and numeric.size == 2:
        outside = (values < numeric[0]) | (values > numeric[1])
        return np.any(outside, axis=0)
    if numeric.shape == values.shape:
        return np.any(values >= numeric, axis=0)
    if numeric.ndim == 2 and numeric.shape == (values.shape[0], 2):
        outside = (values < numeric[:, :1]) | (values > numeric[:, 1:])
        return np.any(outside, axis=0)
    if numeric.ndim == 3 and numeric.shape[:2] == values.shape and numeric.shape[2] == 2:
        outside = (values < numeric[:, :, 0]) | (values > numeric[:, :, 1])
        return np.any(outside, axis=0)
    if numeric.ndim == 1 and numeric.size in {1, values.shape[0]}:
        threshold = np.broadcast_to(numeric.reshape(-1, 1), values.shape)
        return np.any(values >= threshold, axis=0)
    raise ValueError("plotcurve maskarray shape does not match data")


def _highlight_regions(ax: Axes, times: np.ndarray, regions: np.ndarray, mode: str, xlabel: str) -> None:
    target = ax
    if mode == "bottom":
        left, bottom, width, height = ax.get_position().bounds
        ax.set_position([left + 0.1 * width, bottom + 0.1 * height, 0.9 * width, 0.85 * height])
        target = ax.figure.add_axes([left + 0.1 * width, bottom + 0.05 * height, 0.9 * width, 0.05 * height])
        target.set_xlim(float(times[0]), float(times[-1]))
        target.set_ylim(0.0, 1.0)
        target.set_yticks([])
        target.set_xlabel(xlabel)
        ax.set_xticks([])
    else:
        ax.set_xlabel(xlabel)
    for start, stop in _true_runs(regions):
        right = min(stop, times.size - 1)
        target.axvspan(times[start], times[right], color="0.75" if mode == "background" else "k", zorder=-1)


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    regions = np.asarray(values, dtype=bool).ravel()
    changes = np.diff(np.pad(regions.astype(int), (1, 1)))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def _y_limits(data: np.ndarray, requested: Any, stderr: Any) -> tuple[float, float]:
    if requested is not None and np.asarray(requested).size:
        values = np.asarray(requested, dtype=float).ravel()
        if values.size == 1:
            return float(values[0]), float(np.nanmax(data))
        if values.size != 2:
            raise ValueError("plotcurve ylim must contain one or two values")
        if np.all(values == 0):
            return 0.0, 1.0
        return float(values[0]), float(values[1])
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if stderr is not None and np.asarray(stderr).size:
        error = float(np.nanmax(np.abs(np.asarray(stderr, dtype=float))))
        low, high = low - error, high + error
    span = high - low
    padding = span / 10.0 if span else max(abs(high) / 10.0, 0.1)
    return low - padding, high + padding


def _plot_topographies(ax: Axes, values: Any, chanlocs: Any, titles: Sequence[str]) -> None:
    if chanlocs is None:
        raise ValueError("plotcurve plottopo requires chanlocs")
    maps = np.asarray(values, dtype=float)
    if maps.ndim == 1:
        maps = maps[np.newaxis, :]
    left, bottom, width, height = ax.get_position().bounds
    ax.set_position([left, bottom, width, height / 2.0])
    for index, map_values in enumerate(maps):
        topo_ax = ax.figure.add_axes(
            [left + index * width / maps.shape[0], bottom + height / 2.0, width / maps.shape[0], height / 2.0]
        )
        topoplot(map_values, chanlocs, axes=topo_ax, colorbar=False)
        if index < len(titles):
            topo_ax.set_title(str(titles[index]))


def _finish_axes(ax: Axes, lines: list[Any], options: dict[str, Any]) -> None:
    if options["plotmode"] == "topo":
        low, high = ax.get_ylim()
        left, right = ax.get_xlim()
        ax.text(left + 0.1 * (right - left), high - 0.2 * (high - low), str(options["title"]))
        ax.axvline(0.0, color="k", linewidth=1)
        ax.axhline(0.0, color="k", linewidth=1)
        ax.set_axis_off()
        setattr(ax, "_eegprep_plotcurve_metadata", (options["xlabel"], options["ylabel"], options["legend"]))
        return
    ax.set_title(str(options["title"]))
    if options["maskarray"] is None or not np.asarray(options["maskarray"]).size:
        ax.set_xlabel(str(options["xlabel"]))
    ax.set_ylabel(str(options["ylabel"]))
    labels = tuple(str(value) for value in options["legend"])
    if labels:
        ax.legend(lines[: len(labels)], labels, loc="lower right")
    if options["logpval"]:
        ticks = ax.get_yticks()
        ax.set_yticks(ticks, np.round(10.0 ** (-ticks) * 1000.0) / 1000.0)
        ax.invert_yaxis()


def _style_parts(value: Any) -> tuple[Any, str]:
    if not isinstance(value, str) or len(value) <= 1:
        return value, "-"
    if value[0] in "rgbcmykw" and value[1:] in {"-", "--", ":", "-."}:
        return value[0], value[1:]
    return value, "-"


def _is_on(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "on"
    return bool(value)


__all__ = ["DEFAULT_CURVE_COLORS", "plotcurve"]

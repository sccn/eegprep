"""Topographic or rectangular ERP array plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list, channel_has_coordinates
from eegprep.functions.sigprocfunc.topoplot import topo_screen_coords


def plottopo(
    data: Any,
    *,
    times: Any = None,
    chanlocs: Any = None,
    title: str = "",
    channels: Any = None,
    ydir: int = -1,
    ylimits: Any = None,
    rect: bool = False,
    singletrials: bool = False,
    overlay: bool = False,
    styles: list[dict[str, Any]] | None = None,
    regions: Any = None,
):
    """Plot channel/component traces in an EEGLAB-like array.

    Three-dimensional input is averaged over trials by default. Set
    ``singletrials=True`` to overlay every trial in each channel panel.
    ``overlay=True`` keeps the third axis as separate styled traces.
    Every panel shares one amplitude range, and a calibration bar shows it.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 3 and not singletrials and not overlay:
        values = np.nanmean(values, axis=2)
    if values.ndim not in {2, 3}:
        raise ValueError("plottopo data must be channels x points or channels x points x trials")
    count, points = values.shape[:2]
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(points)
    )
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    indices = _indices(channels, count)
    labels = _labels(chanlocs, count)
    limits = _shared_limits(values, indices, _limits(ylimits))
    region_values = None if regions is None else np.asarray(regions, dtype=bool)
    if not rect:
        positions = _topographic_positions(chanlocs, indices)
        if positions is not None:
            return _topographic_figure(
                x_values,
                values,
                labels,
                indices,
                positions,
                title,
                ydir,
                limits,
                styles=styles,
                regions=region_values,
            )
    rows, cols = _grid_shape(len(indices))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(cols * 2.1, rows * 1.6))
    for ax, index in zip(axes.ravel(), indices):
        channel_region = None if region_values is None else region_values[int(index)]
        _plot_trace(
            ax,
            x_values,
            values[index],
            labels[index],
            ydir,
            limits,
            styles=styles,
            region=channel_region,
        )
    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _add_scale_bar(fig, x_values, limits, ydir)
    _add_legend(fig, styles)
    return fig


def _grid_shape(count: int) -> tuple[int, int]:
    cols = int(np.ceil(np.sqrt(max(count, 1))))
    rows = int(np.ceil(max(count, 1) / cols))
    return rows, cols


def _indices(channels: Any, count: int) -> np.ndarray:
    if channels is None or (isinstance(channels, (list, tuple)) and not channels):
        return np.arange(count)
    values = np.asarray(channels, dtype=int).ravel()
    if values.size == 0:
        return np.arange(count)
    if np.any(values < 1) or np.any(values > count):
        raise ValueError(f"channels must be 1-based and within 1..{count}")
    return values - 1


def _labels(chanlocs: Any, count: int) -> list[str]:
    labels = [str(loc.get("labels") or index) for index, loc in enumerate(chanlocs_as_list(chanlocs), start=1)]
    while len(labels) < count:
        labels.append(str(len(labels) + 1))
    return labels


def _topographic_positions(chanlocs: Any, indices: np.ndarray) -> np.ndarray | None:
    locs = chanlocs_as_list(chanlocs)
    if not locs or np.max(indices, initial=-1) >= len(locs):
        return None
    if not any(channel_has_coordinates(locs[int(index)]) for index in indices):
        return None
    side_rows = int(np.floor(np.sqrt(max(len(indices), 1)))) + 1
    empty_seen = 0
    positions = []
    for index in indices:
        loc = locs[int(index)]
        if channel_has_coordinates(loc):
            theta = float(np.asarray(loc["theta"], dtype=float).reshape(-1)[0])
            radius = float(np.asarray(loc["radius"], dtype=float).reshape(-1)[0])
            positions.append(topo_screen_coords(theta, radius))
            continue
        # EEGLAB plottopo parks channels with an empty theta beside the head.
        positions.append(
            (
                0.7 + 0.2 * np.floor(empty_seen / side_rows),
                -0.4 + (empty_seen % side_rows) / max(side_rows, 1),
            )
        )
        empty_seen += 1
    return np.asarray(positions, dtype=float)


def _topographic_figure(
    x_values: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    indices: np.ndarray,
    positions: np.ndarray,
    title: str,
    ydir: int,
    limits: tuple[float, float],
    *,
    styles: list[dict[str, Any]] | None,
    regions: np.ndarray | None,
):
    fig = plt.figure(figsize=(7.2, 5.8))
    if title:
        fig.suptitle(title, fontweight="bold")
    axis_width = min(0.18, max(0.09, 0.72 / np.sqrt(max(len(indices), 1)) / 1.7))
    axis_height = axis_width * 0.78
    scale = 0.62
    for index, (x_pos, y_pos) in zip(indices, positions):
        left = 0.46 + float(x_pos) * scale - axis_width / 2
        bottom = 0.50 + float(y_pos) * scale - axis_height / 2
        left = float(np.clip(left, 0.03, 0.82 - axis_width))
        bottom = float(np.clip(bottom, 0.08, 0.92 - axis_height))
        ax = fig.add_axes([left, bottom, axis_width, axis_height])
        channel_region = None if regions is None else regions[int(index)]
        _plot_trace(
            ax,
            x_values,
            values[index],
            labels[index],
            ydir,
            limits,
            styles=styles,
            region=channel_region,
        )
    _add_scale_bar(fig, x_values, limits, ydir)
    _add_legend(fig, styles)
    return fig


def _plot_trace(
    ax: Any,
    x_values: np.ndarray,
    values: np.ndarray,
    label: str,
    ydir: int,
    limits: tuple[float, float],
    *,
    styles: list[dict[str, Any]] | None,
    region: np.ndarray | None,
) -> None:
    series = np.asarray(values, dtype=float)
    if styles:
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        for trace_index, style in enumerate(styles):
            column = series[:, trace_index] if series.ndim == 2 else series
            plotted = {key: style[key] for key in ("color", "linestyle", "linewidth") if key in style}
            if style.get("label"):
                plotted["label"] = style["label"]
            ax.plot(x_values, column, **plotted)
    else:
        ax.plot(x_values, series, color="black", linewidth=0.8)
    ax.axhline(0, color="0.75", linewidth=0.6)
    if region is not None:
        _shade_region(ax, x_values, np.asarray(region, dtype=bool).ravel())
    ax.set_title(label, fontsize=9)
    ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    ax.set_ylim(limits)
    if ydir < 0:
        ax.invert_yaxis()
    ax.tick_params(labelsize=7)


def _shade_region(ax: Any, x_values: np.ndarray, region: np.ndarray) -> None:
    if region.size != x_values.size:
        return
    starts: list[int] = []
    stops: list[int] = []
    active = False
    for index, marked in enumerate(region.tolist()):
        if marked and not active:
            starts.append(index)
            active = True
        elif not marked and active:
            stops.append(index - 1)
            active = False
    if active:
        stops.append(int(region.size) - 1)
    for start, stop in zip(starts, stops):
        ax.axvspan(float(x_values[start]), float(x_values[stop]), color="gold", alpha=0.18, linewidth=0)


def _add_scale_bar(fig: Any, x_values: np.ndarray, limits: tuple[float, float], ydir: int) -> None:
    ax = fig.add_axes([0.84, 0.12, 0.12, 0.28])
    ax.set_title("")
    xmin = float(x_values[0])
    xmax = float(x_values[-1])
    ymin, ymax = limits
    origin = 0.0 if xmin <= 0 <= xmax else xmin
    ax.plot([origin, origin], [ymin, ymax], color="black", linewidth=1.0)
    ax.plot([xmin, xmax], [0, 0], color="0.45", linewidth=0.6)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if ydir < 0:
        ax.invert_yaxis()
    ax.text(origin, ymin, _scale_text(ymin), fontsize=8, ha="right", va="top")
    positive = _scale_text(ymax)
    ax.text(origin, ymax, positive if positive.startswith("-") else f"+{positive}", fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("Time (ms)", fontsize=8)
    ax.tick_params(labelleft=False, labelbottom=False, length=0)
    ax.set_xticks([xmin, xmax])
    ax.set_xticklabels([_scale_text(xmin), _scale_text(xmax)], fontsize=7)


def _add_legend(fig: Any, styles: list[dict[str, Any]] | None) -> None:
    if not styles:
        return
    handles = []
    labels = []
    seen: set[str] = set()
    for style in styles:
        label = style.get("label")
        if not label or label in seen:
            continue
        seen.add(str(label))
        handles.append(
            plt.Line2D(
                [],
                [],
                color=style.get("color", "black"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.2),
            )
        )
        labels.append(str(label))
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)


def _scale_text(value: float) -> str:
    return f"{float(value):.3g}"


def _shared_limits(values: np.ndarray, indices: np.ndarray, limits: tuple[float, float] | None) -> tuple[float, float]:
    if limits is not None:
        return limits
    selected = values[indices]
    finite = selected[np.isfinite(selected)]
    peak = float(np.max(np.abs(finite))) if finite.size else 1.0
    if peak == 0:
        peak = 1.0
    return (-peak, peak)


def _limits(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float).ravel()
    if values.size != 2 or np.all(values == 0):
        return None
    return float(values[0]), float(values[1])


__all__ = ["plottopo"]

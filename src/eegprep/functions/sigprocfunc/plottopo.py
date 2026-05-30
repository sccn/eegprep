"""Topographic or rectangular ERP array plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


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
):
    """Plot channel/component traces in an EEGLAB-like array."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        values = np.nanmean(values, axis=2)
    if values.ndim != 2:
        raise ValueError("plottopo data must be channels x points")
    count, points = values.shape
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(points)
    )
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    indices = _indices(channels, count)
    labels = _labels(chanlocs, count)
    limits = _limits(ylimits)
    if not rect:
        positions = _topographic_positions(chanlocs, indices)
        if positions is not None:
            fig = _topographic_figure(x_values, values, labels, indices, positions, title, ydir, limits)
            return fig
    rows, cols = _grid_shape(len(indices))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(cols * 2.1, rows * 1.6))
    for ax, index in zip(axes.ravel(), indices):
        _plot_trace(ax, x_values, values[index], labels[index], ydir, limits)
    for ax in axes.ravel()[len(indices) :]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
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
    positions = []
    for index in indices:
        loc = locs[int(index)]
        try:
            theta = np.deg2rad(float(loc["theta"]))
            radius = float(loc["radius"])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(theta) or not np.isfinite(radius):
            return None
        positions.append((-np.sin(theta) * radius, np.cos(theta) * radius))
    return np.asarray(positions, dtype=float)


def _topographic_figure(
    x_values: np.ndarray,
    values: np.ndarray,
    labels: list[str],
    indices: np.ndarray,
    positions: np.ndarray,
    title: str,
    ydir: int,
    limits: tuple[float, float] | None,
):
    fig = plt.figure(figsize=(7.0, 5.8))
    if title:
        fig.suptitle(title, fontweight="bold")
    axis_width = min(0.22, max(0.10, 0.86 / np.sqrt(max(len(indices), 1)) / 1.7))
    axis_height = axis_width * 0.78
    scale = 0.80
    for index, (x_pos, y_pos) in zip(indices, positions):
        left = 0.5 + x_pos * scale - axis_width / 2
        bottom = 0.48 + y_pos * scale - axis_height / 2
        left = float(np.clip(left, 0.04, 0.96 - axis_width))
        bottom = float(np.clip(bottom, 0.04, 0.92 - axis_height))
        ax = fig.add_axes([left, bottom, axis_width, axis_height])
        _plot_trace(ax, x_values, values[index], labels[index], ydir, limits)
    return fig


def _plot_trace(
    ax: Any,
    x_values: np.ndarray,
    values: np.ndarray,
    label: str,
    ydir: int,
    limits: tuple[float, float] | None,
) -> None:
    ax.plot(x_values, values, color="black", linewidth=0.8)
    ax.axhline(0, color="0.75", linewidth=0.6)
    ax.set_title(label, fontsize=9)
    if ydir < 0:
        ax.invert_yaxis()
    if limits is not None:
        ax.set_ylim(limits)
    ax.tick_params(labelsize=7)


def _limits(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float).ravel()
    if values.size != 2 or np.all(values == 0):
        return None
    return float(values[0]), float(values[1])


__all__ = ["plottopo"]

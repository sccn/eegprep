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
    rows, cols = _grid_shape(len(indices))
    fig, axes = plt.subplots(rows, cols, squeeze=False, figsize=(cols * 2.1, rows * 1.6))
    for ax, index in zip(axes.ravel(), indices):
        ax.plot(x_values, values[index], color="black", linewidth=0.8)
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.set_title(labels[index], fontsize=9)
        if ydir < 0:
            ax.invert_yaxis()
        ax.tick_params(labelsize=7)
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


__all__ = ["plottopo"]

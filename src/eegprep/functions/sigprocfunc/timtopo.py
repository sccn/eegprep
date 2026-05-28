"""ERP trace plus scalp map helper matching EEGLAB ``timtopo`` basics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot


def timtopo(
    data: Any,
    chanlocs: Any,
    *,
    times: Any = None,
    plottimes: Any = None,
    winsize: float = 0.0,
    title: str = "",
    topoplot_options: dict[str, Any] | None = None,
):
    """Plot all channel traces and scalp maps at selected latencies."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        values = np.nanmean(values, axis=2)
    if values.ndim != 2:
        raise ValueError("timtopo data must be channels x points")
    points = values.shape[1]
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(points)
    )
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    map_times = _plot_times(values, x_values, plottimes)
    fig = plt.figure(figsize=(8, 4 + 1.7 * max(1, int(np.ceil(len(map_times) / 4)))))
    trace_ax = fig.add_subplot(2, 1, 1)
    trace_ax.plot(x_values, values.T, linewidth=0.7)
    trace_ax.axhline(0, color="0.7", linewidth=0.6)
    for latency in map_times:
        trace_ax.axvline(latency, color="black", linestyle=":", linewidth=0.8)
    trace_ax.set_xlabel("Time (ms)")
    trace_ax.set_ylabel("uV")
    trace_ax.set_title(title or "Channel ERPs with scalp maps")
    for index, latency in enumerate(map_times, start=1):
        ax = fig.add_subplot(2, max(len(map_times), 1), max(len(map_times), 1) + index)
        map_values = _latency_values(values, x_values, latency, winsize)
        topoplot(map_values, chanlocs_as_list(chanlocs), axes=ax, electrodes="off", **(topoplot_options or {}))
        ax.set_title(f"{latency:g} ms")
    fig.tight_layout()
    return fig


def _plot_times(values: np.ndarray, x_values: np.ndarray, plottimes: Any) -> np.ndarray:
    if plottimes is not None and len(np.asarray(plottimes).ravel()):
        requested = np.asarray(plottimes, dtype=float).ravel()
        requested = requested[np.isfinite(requested)]
        if requested.size:
            return requested
    variance = np.nanvar(values, axis=0)
    return np.asarray([x_values[int(np.nanargmax(variance))]], dtype=float)


def _latency_values(values: np.ndarray, x_values: np.ndarray, latency: float, winsize: float) -> np.ndarray:
    if winsize <= 0:
        frame = int(np.argmin(np.abs(x_values - latency)))
        return values[:, frame]
    half_window = winsize / 2.0
    mask = (x_values >= latency - half_window) & (x_values <= latency + half_window)
    if not np.any(mask):
        raise ValueError("winsize does not include any samples around requested latency")
    return np.nanmean(values[:, mask], axis=1)


__all__ = ["timtopo"]

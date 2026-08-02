"""ERP trace plus scalp map helper matching EEGLAB ``timtopo``."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot

# MATLAB's default axes color order, cycled across the oblique leader lines as
# EEGLAB ``timtopo`` does when it draws each leader with no explicit color.
_LEADER_COLORS = [
    (0.000, 0.447, 0.741),
    (0.850, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
    (0.466, 0.674, 0.188),
    (0.301, 0.745, 0.933),
    (0.635, 0.078, 0.184),
]


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
    """Plot all channel traces and scalp maps at selected latencies.

    The layout mirrors EEGLAB ``timtopo``: a row of scalp maps sits above an ERP
    trace panel, each map joined to a blue latency marker on the traces by an
    oblique leader line, with a ``+``/``-`` polarity colorbar on the right. Clicking
    a trace redraws the rightmost scalp map at the clicked latency (interactive
    backends only).
    """
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
    locs = chanlocs_as_list(chanlocs)
    draw_maps = bool(locs)

    if draw_maps:
        fig = plt.figure(figsize=(8, 6))
        trace_ax = fig.add_axes([0.13, 0.10, 0.80, 0.50])
    else:
        fig, trace_ax = plt.subplots(figsize=(8, 4))

    trace_ax.plot(x_values, values.T, linewidth=0.7)
    trace_ax.set_xlabel("Latency (ms)")
    trace_ax.set_ylabel("Potential (µV)")
    trace_ax.grid(True, axis="y", linestyle=":")
    trace_ax.set_xlim(float(x_values[0]), float(x_values[-1]))
    y_low, y_high = float(np.nanmin(values)), float(np.nanmax(values))
    if y_high > y_low:
        trace_ax.set_ylim(y_low, y_high)
    if x_values[0] < 0 < x_values[-1]:
        trace_ax.axvline(0, color="k", linestyle=":", linewidth=1.5)

    if draw_maps:
        target_ax, plot_options, target_leader = _draw_maps_row(
            fig, trace_ax, values, x_values, map_times, locs, winsize, topoplot_options
        )
        _enable_click_maps(fig, trace_ax, target_ax, target_leader, values, x_values, locs, winsize, plot_options)
    else:
        for latency in map_times:
            trace_ax.axvline(latency, color="b", linewidth=1.0)

    if title:
        # EEGLAB places the timtopo title between the trace panel and the maps, left-aligned.
        fig.text(0.03, 0.62, title, ha="left", fontsize=12)
    return fig


def _draw_maps_row(
    fig: Any,
    trace_ax: Any,
    values: np.ndarray,
    x_values: np.ndarray,
    map_times: np.ndarray,
    locs: list,
    winsize: float,
    topoplot_options: dict[str, Any] | None,
) -> tuple[Any, dict[str, Any], Any]:
    """Draw the top row of scalp maps, the blue latency markers with oblique leader
    lines down to the traces, and the ``+``/``-`` polarity colorbar.

    Each map is scaled independently (``maplimits='absmax'``, EEGLAB's default), so
    the shared colorbar is polarity-only, not a common data scale. Returns the
    rightmost map axes, the topoplot options, and that map's leader line, which the
    click callback reuses."""
    count = len(map_times)
    top_y, top_h = 0.66, 0.26
    left, right = 0.10, 0.88
    slot = (right - left) / count
    map_w = min(slot * 0.92, 0.24)
    # EEGLAB shows electrode markers unless the maps get tiny (topowidth < 0.12).
    plot_options = {"electrodes": "on", "maplimits": "absmax", **(topoplot_options or {})}
    if map_w < 0.12:
        plot_options["electrodes"] = "off"

    for index, latency in enumerate(map_times):
        frame = int(np.argmin(np.abs(x_values - latency)))
        map_values = _latency_values(values, x_values, latency, winsize)
        center = left + slot * (index + 0.5)
        topo_ax = fig.add_axes([center - map_w / 2, top_y, map_w, top_h])
        topoplot(map_values, locs, axes=topo_ax, **plot_options)
        topo_ax.set_title(f"{latency:.0f}", fontweight="bold", fontsize=10)

        # Blue vertical line through the data range at this latency, over a white
        # underlay so it reads clearly against the multicolored traces (EEGLAB).
        column = values[:, frame]
        v_low, v_high = float(np.nanmin(column)), float(np.nanmax(column))
        trace_ax.plot([latency, latency], [v_low, v_high], color="w", linewidth=2.0, zorder=3)
        trace_ax.plot([latency, latency], [v_low, v_high], color="b", linewidth=1.5, zorder=4)

        # Oblique leader line from the marker's top to the bottom-center of the map,
        # cycling MATLAB's default color order the way EEGLAB does.
        leader = fig.add_artist(
            ConnectionPatch(
                xyA=(latency, v_high),
                coordsA=trace_ax.transData,
                xyB=(0.5, 0.0),
                coordsB=topo_ax.transAxes,
                color=_LEADER_COLORS[index % len(_LEADER_COLORS)],
                linewidth=1.0,
            )
        )

    cbar_ax = fig.add_axes([0.925, top_y + 0.07, 0.018, top_h - 0.14])
    cmap = plt.get_cmap((topoplot_options or {}).get("colormap") or "turbo")
    colorbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax)
    colorbar.set_ticks([-0.8, 0, 0.8])
    colorbar.set_ticklabels(["-", "", "+"])
    colorbar.ax.tick_params(length=0)
    return topo_ax, plot_options, leader


def _enable_click_maps(
    fig: Any,
    trace_ax: Any,
    target_ax: Any,
    target_leader: Any,
    values: np.ndarray,
    x_values: np.ndarray,
    locs: list,
    winsize: float,
    plot_options: dict[str, Any],
) -> None:
    """Redraw the rightmost scalp map at the clicked ERP latency, matching EEGLAB
    timtopo's click callback. Fires only on interactive backends; headless renders
    receive no click events, so this is a harmless no-op."""
    srate = (values.shape[1] - 1) / (float(x_values[-1]) - float(x_values[0])) * 1000.0

    def _on_click(event: Any) -> None:
        if event.inaxes is not trace_ax or event.xdata is None:
            return
        latency = float(event.xdata)
        # Hide the rightmost map's static leader; it no longer points at the redrawn map.
        target_leader.set_visible(False)
        target_ax.clear()
        topoplot(_click_latency_values(values, x_values, srate, latency, winsize), locs, axes=target_ax, **plot_options)
        if winsize and winsize > 0:
            label = f"{latency - winsize:.0f} to {latency + winsize:.0f} ms"
        else:
            label = f"{latency:.0f} ms"
        target_ax.set_title(label, fontweight="bold", fontsize=10)
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)


def _click_latency_values(
    values: np.ndarray, x_values: np.ndarray, srate: float, latency: float, winsize: float
) -> np.ndarray:
    """ERP averaged over +/- winsize ms around a clicked latency (single frame when
    winsize == 0), matching EEGLAB timtopo's click callback."""
    center = int(round((latency - float(x_values[0])) / 1000.0 * srate))
    winpts = int(round(winsize / 1000.0 * srate))
    lo = max(0, center - winpts)
    hi = min(values.shape[1] - 1, center + winpts)
    return np.nanmean(values[:, lo : hi + 1], axis=1)


def _plot_times(values: np.ndarray, x_values: np.ndarray, plottimes: Any) -> np.ndarray:
    if plottimes is not None and len(np.asarray(plottimes).ravel()):
        requested = np.asarray(plottimes, dtype=float).ravel()
        requested = requested[np.isfinite(requested)]
        if requested.size:
            return requested
    # Default frame is the peak global power (sum of squares across channels), matching
    # EEGLAB timtopo's ``max(sum(data.*data))`` -- not the mean-removed variance.
    power = np.nansum(values**2, axis=0)
    return np.asarray([x_values[int(np.nanargmax(power))]], dtype=float)


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

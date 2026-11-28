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
    limits: Any = None,
    plottimes: Any = None,
    winsize: float = 0.0,
    plotchans: Any = None,
    voffsets: Any = None,
    plotenvelope: Any = 0,
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
    plot_xlim, plot_ylim = _parse_limits(limits)
    if times is not None and len(np.asarray(times).ravel()):
        x_values = np.asarray(times, dtype=float).ravel()
    elif plot_xlim is not None:
        x_values = np.linspace(plot_xlim[0], plot_xlim[1], points)
    else:
        x_values = np.arange(points, dtype=float)
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    trace_chans = _resolve_plotchans(plotchans, values.shape[0])
    trace_values = values[trace_chans]
    map_times = _plot_times(values, x_values, plottimes)
    resolved_voffsets = _resolve_voffsets(voffsets, map_times.size)
    locs = chanlocs_as_list(chanlocs)
    draw_maps = bool(locs)

    if draw_maps:
        fig = plt.figure(figsize=(8, 6))
        trace_ax = fig.add_axes([0.13, 0.10, 0.80, 0.50])
    else:
        fig, trace_ax = plt.subplots(figsize=(8, 4))

    trace_ax.plot(x_values, trace_values.T, linewidth=0.7)
    if _truthy_flag(plotenvelope):
        trace_ax.plot(x_values, np.nanmax(values, axis=0), color="k", linewidth=2.0, zorder=3)
        trace_ax.plot(x_values, np.nanmin(values, axis=0), color="k", linewidth=2.0, zorder=3)
    trace_ax.set_xlabel("Latency (ms)")
    trace_ax.set_ylabel("Potential (µV)")
    trace_ax.grid(True, axis="y", linestyle=":")
    x_low, x_high = plot_xlim if plot_xlim is not None else (float(x_values[0]), float(x_values[-1]))
    trace_ax.set_xlim(x_low, x_high)
    y_low, y_high = (
        plot_ylim if plot_ylim is not None else (float(np.nanmin(trace_values)), float(np.nanmax(trace_values)))
    )
    if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
        trace_ax.set_ylim(y_low, y_high)
    if x_low < 0 < x_high:
        trace_ax.axvline(0, color="k", linestyle=":", linewidth=1.5)

    if draw_maps:
        target_ax, plot_options, target_leader = _draw_maps_row(
            fig,
            trace_ax,
            values,
            trace_values,
            x_values,
            map_times,
            locs,
            winsize,
            resolved_voffsets,
            topoplot_options,
        )
        _enable_click_maps(fig, trace_ax, target_ax, target_leader, values, x_values, locs, winsize, plot_options)
    else:
        for latency in map_times:
            trace_ax.axvline(latency, color="b", linewidth=1.0)

    if title:
        if draw_maps:
            # EEGLAB places the timtopo title between the trace panel and the maps, left-aligned.
            fig.text(0.03, 0.62, title, ha="left", fontsize=12)
        else:
            # No maps row to sit under; keep the title on the trace panel instead of at y=0.62,
            # which would land inside the full-height axes.
            trace_ax.set_title(title)
    return fig


def _draw_maps_row(
    fig: Any,
    trace_ax: Any,
    values: np.ndarray,
    trace_values: np.ndarray,
    x_values: np.ndarray,
    map_times: np.ndarray,
    locs: list,
    winsize: float,
    voffsets: np.ndarray,
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
        column = trace_values[:, frame]
        v_low, v_high = float(np.nanmin(column)), float(np.nanmax(column))
        v_top = v_high + (float(voffsets[index]) if index < voffsets.size else 0.0)
        trace_ax.plot([latency, latency], [v_low, v_top], color="w", linewidth=2.0, zorder=3)
        trace_ax.plot([latency, latency], [v_low, v_top], color="b", linewidth=1.5, zorder=4)

        # Oblique leader line from the marker's top to the bottom-center of the map,
        # cycling MATLAB's default color order the way EEGLAB does.
        leader = fig.add_artist(
            ConnectionPatch(
                xyA=(latency, v_top),
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
    # Inset the +/- labels from the bar ends (EEGLAB places them at the outermost ticks,
    # not flush with the extremes). Purely cosmetic; the bar spans the full gradient and
    # does not drive the per-map scalp colors.
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

    def _on_click(event: Any) -> None:
        if event.inaxes is not trace_ax or event.xdata is None:
            return
        latency = float(event.xdata)
        # Hide the rightmost map's static leader; it no longer points at the redrawn map.
        target_leader.set_visible(False)
        target_ax.clear()
        topoplot(_latency_values(values, x_values, latency, winsize), locs, axes=target_ax, **plot_options)
        if winsize and winsize > 0:
            label = f"{latency - winsize:.0f} to {latency + winsize:.0f} ms"
        else:
            label = f"{latency:.0f} ms"
        target_ax.set_title(label, fontweight="bold", fontsize=10)
        event.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)


def _plot_times(values: np.ndarray, x_values: np.ndarray, plottimes: Any) -> np.ndarray:
    # Empty/NaN slots use the peak global power latency (sum of squares across channels),
    # matching EEGLAB timtopo's ``max(sum(data.*data))`` -- not the mean-removed variance.
    auto = float(x_values[int(np.nanargmax(np.nansum(values**2, axis=0)))])
    requested = np.asarray(plottimes, dtype=float).ravel() if plottimes is not None else np.asarray([], dtype=float)
    if requested.size == 0:
        return np.asarray([auto], dtype=float)
    # EEGLAB replaces each NaN entry with the peak-power latency, keeping the finite
    # entries and the slot count, rather than dropping the NaN slot.
    requested = requested.copy()
    requested[np.isnan(requested)] = auto
    return requested


def _latency_values(values: np.ndarray, x_values: np.ndarray, latency: float, winsize: float) -> np.ndarray:
    """ERP at ``latency``: a single frame when ``winsize == 0``, otherwise averaged over
    ``latency +/- winsize`` ms. This is EEGLAB timtopo's half-width convention, shared by
    the static map row and the click callback so both use one window for a given input."""
    if winsize <= 0:
        frame = int(np.argmin(np.abs(x_values - latency)))
        return values[:, frame]
    mask = (x_values >= latency - winsize) & (x_values <= latency + winsize)
    if not np.any(mask):
        raise ValueError("winsize does not include any samples around requested latency")
    return np.nanmean(values[:, mask], axis=1)


def _parse_limits(limits: Any) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    if limits is None:
        return None, None
    vals = np.asarray(limits, dtype=float).ravel()
    if vals.size == 0 or (vals.size == 1 and vals[0] == 0.0):
        return None, None
    if vals.size not in (2, 4):
        raise ValueError("timtopo: limits must be 0, [minms maxms], or [minms maxms minval maxval]")
    plot_xlim = None
    if np.isfinite(vals[0]) and np.isfinite(vals[1]) and (vals[0] != 0.0 or vals[1] != 0.0):
        if vals[1] <= vals[0]:
            raise ValueError("timtopo: in limits, maxms must be > minms")
        plot_xlim = (float(vals[0]), float(vals[1]))
    plot_ylim = None
    if vals.size == 4 and np.isfinite(vals[2]) and np.isfinite(vals[3]) and (vals[2] != 0.0 or vals[3] != 0.0):
        if vals[3] <= vals[2]:
            raise ValueError("timtopo: in limits, maxval must be > minval")
        plot_ylim = (float(vals[2]), float(vals[3]))
    return plot_xlim, plot_ylim


def _resolve_plotchans(plotchans: Any, n_chans: int) -> np.ndarray:
    if plotchans is None:
        return np.arange(n_chans, dtype=int)
    vals = np.asarray(plotchans, dtype=int).ravel()
    if vals.size == 0 or (vals.size == 1 and vals[0] == 0):
        return np.arange(n_chans, dtype=int)
    indices = vals - 1
    if np.any(indices < 0) or np.any(indices >= n_chans):
        raise ValueError(f"timtopo: plotchans must be 1-based indices in 1..{n_chans}")
    return indices


def _resolve_voffsets(voffsets: Any, n_topos: int) -> np.ndarray:
    if n_topos <= 0 or voffsets is None:
        return np.zeros(max(n_topos, 0), dtype=float)
    vals = np.asarray(voffsets, dtype=float).ravel()
    if vals.size == 0 or (vals.size == 1 and vals[0] == 0.0):
        return np.zeros(n_topos, dtype=float)
    return np.resize(vals, n_topos)


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"on", "yes", "true", "1"}
    arr = np.asarray(value).ravel()
    return bool(arr.size and arr[0])


__all__ = ["timtopo"]

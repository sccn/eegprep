"""Envelope of a data epoch plus scalp maps of the largest components.

Port of EEGLAB ``envtopo``. Ranks ICA components by their back-projected
contribution to the (mean) data epoch, plots the data envelope with the summed
and per-component contribution envelopes, and draws the scalp map of each of the
largest contributors.

``envtopo`` returns the six EEGLAB contribution outputs plus the figure as an
``EnvtopoResult``. Two deliberate departures from the MATLAB code, neither
observable in the ranked results:

* ``compvars`` holds the sort metric in ranked (descending) order. MATLAB
  returns the per-component maximum back-projected power simply reversed, which
  is neither sorted nor mode-dependent; the ranked metric is the useful value.
* the ``'rms'`` envelope divides by the per-sample count of positive/negative
  channels. MATLAB's ``'rms'`` branch references undefined variables and errors.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot

MAXTOPOS = 20  # EEGLAB caps the number of plotted component maps
_FILLCOLOR = (0.815, 0.94, 1.0)  # light blue summed-projection fill, as EEGLAB
# Deprecated EEGLAB sortvar spellings kept for parity with older scripts.
_SORTVAR_ALIASES = {"on": "mp", "pvaf": "mp", "mv": "mp", "off": "rp", "rv": "rp"}
_VALID_SORTVAR = ("mp", "pv", "pp", "rp")
# MATLAB default axes color order, cycled across component lines / leaders.
_COMP_COLORS = [
    (0.850, 0.325, 0.098),
    (0.929, 0.694, 0.125),
    (0.494, 0.184, 0.556),
    (0.466, 0.674, 0.188),
    (0.301, 0.745, 0.933),
    (0.635, 0.078, 0.184),
    (0.000, 0.447, 0.741),
]


class EnvtopoResult(NamedTuple):
    """Component-contribution outputs of :func:`envtopo` plus the figure.

    ``compvarorder``/``compsplotted`` are 1-based component numbers (EEGLAB
    facing); ``compframes`` are 0-based sample indices; ``comptimes`` are in ms.
    """

    compvarorder: np.ndarray
    compvars: np.ndarray
    compframes: np.ndarray
    comptimes: np.ndarray
    compsplotted: np.ndarray
    sortvar: np.ndarray
    figure: Figure


def envtopo(
    data: Any,
    weights: Any,
    *,
    chanlocs: Any = None,
    icawinv: Any = None,
    timerange: Any = None,
    limcontrib: Any = None,
    compnums: Any = None,
    compsplot: int = 7,
    subcomps: Any = 0,
    sortvar: str = "mp",
    envmode: str = "avg",
    plotchans: Any = None,
    title: str = "",
    topoplot_options: dict[str, Any] | None = None,
) -> EnvtopoResult:
    """Rank component contributions to a data epoch and plot their envelopes/maps.

    Args:
        data: Channels x frames, or channels x frames x epochs (epochs averaged).
        weights: ICA unmixing matrix ``icaweights @ icasphere`` (components x channels).
        chanlocs: Channel locations for the scalp maps, or None to skip the maps.
        icawinv: ICA mixing matrix (channels x components); defaults to ``pinv(weights)``.
        timerange: ``[min max]`` epoch latencies in ms used to label the time axis.
        limcontrib: ``[min max]`` ms window in which contributions are ranked.
        compnums: 1-based candidate components to rank; defaults to all.
        compsplot: Number of largest contributors to plot (capped at 20).
        subcomps: 1-based components to remove before plotting; 0 removes none,
            ``[]`` removes all but ``compnums``.
        sortvar: Ranking metric ``'mp'`` (default), ``'pv'``, ``'pp'`` or ``'rp'``.
        envmode: ``'avg'`` (max/min envelope) or ``'rms'``.
        plotchans: 1-based channels used for envelopes and maps; defaults to all.
        title: Figure title.
        topoplot_options: Extra keyword arguments forwarded to :func:`topoplot`.

    Returns:
        EnvtopoResult: ranking outputs and the matplotlib figure.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        values = np.nanmean(values, axis=2)
    if values.ndim != 2:
        raise ValueError("envtopo data must be channels x frames")
    n_chans, n_frames = values.shape

    weight_matrix = np.asarray(weights, dtype=float)
    if weight_matrix.shape[1] != n_chans:
        raise ValueError("weights columns must match the number of data channels")
    n_components = weight_matrix.shape[0]

    maps = np.linalg.pinv(weight_matrix) if _is_empty(icawinv) else np.asarray(icawinv, dtype=float)
    metric_mode = _normalize_sortvar(sortvar)
    times_ms = _time_axis(timerange, n_frames)
    plot_channels = _resolve_channels(plotchans, n_chans)
    candidates = _resolve_components(compnums, n_components)
    removed = _resolve_subcomps(subcomps, n_components, candidates)

    activations = finite_matmul(weight_matrix, values)
    if removed.size:
        activations[removed, :] = 0.0
    lim1, lim2 = _limcontrib_frames(limcontrib, times_ms, n_frames)

    metric, plotframes, comp_envelopes, max_projections = _contributions(
        values, activations, maps, candidates, plot_channels, lim1, lim2, metric_mode, envmode
    )

    order = np.argsort(metric)[::-1]
    compvarorder = candidates[order] + 1
    compvars = metric[order]
    compframes = plotframes[order]
    comptimes = times_ms[compframes]
    n_topos = min(int(compsplot), candidates.size, MAXTOPOS)
    compsplotted = compvarorder[:n_topos]

    plotted_components = candidates[order][:n_topos]
    figure = _build_figure(
        times_ms=times_ms,
        data_env=_envelope(finite_matmul(maps[plot_channels, :], activations), envmode),
        summed_env=_envelope(
            finite_matmul(maps[np.ix_(plot_channels, plotted_components)], activations[plotted_components]), envmode
        ),
        comp_envelopes=comp_envelopes[order][:n_topos],
        max_projections=max_projections[:, order][:, :n_topos],
        plotted_frames=compframes[:n_topos],
        plotted_labels=compsplotted,
        plot_channels=plot_channels,
        chanlocs=chanlocs,
        limcontrib_ms=None if (lim1, lim2) == (0, n_frames - 1) else (times_ms[lim1], times_ms[lim2]),
        envmode=envmode,
        title=title,
        topoplot_options=topoplot_options,
    )
    return EnvtopoResult(compvarorder, compvars, compframes, comptimes, compsplotted, metric, figure)


def _contributions(values, activations, maps, candidates, plot_channels, lim1, lim2, metric_mode, envmode):
    """Per-candidate ranking metric, peak frame, envelope and peak-frame map."""
    window = slice(lim1, lim2 + 1)
    data_win = values[plot_channels, window]
    reference = float(np.mean(np.var(data_win, axis=0))) if metric_mode == "pv" else float(np.mean(data_win**2))

    metric = np.empty(candidates.size)
    plotframes = np.empty(candidates.size, dtype=int)
    envelopes = np.empty((candidates.size, 2, values.shape[1]))
    max_projections = np.empty((values.shape[0], candidates.size))

    for position, comp in enumerate(candidates):
        projection = np.outer(maps[:, comp], activations[comp, :])
        proj_plot = projection[plot_channels, :]
        proj_win = proj_plot[:, window]

        power = np.mean(proj_win**2, axis=0) if plot_channels.size > 1 else np.abs(proj_win[0, :])
        peak = int(np.argmax(power))
        plotframes[position] = peak + lim1
        max_projections[:, position] = projection[:, peak + lim1]
        envelopes[position] = _envelope(proj_plot, envmode)
        metric[position] = _sort_metric(metric_mode, float(power[peak]), data_win, proj_win, reference)
    return metric, plotframes, envelopes, max_projections


def _sort_metric(metric_mode, max_power, data_win, proj_win, reference):
    if metric_mode == "mp":
        return max_power
    if metric_mode == "pv":
        return 100.0 - 100.0 * float(np.mean(np.var(data_win - proj_win, axis=0))) / reference
    if metric_mode == "pp":
        return 100.0 - 100.0 * float(np.mean((data_win - proj_win) ** 2)) / reference
    return 100.0 * float(np.mean(proj_win**2)) / reference  # "rp"


def _envelope(data, envmode):
    if data.shape[0] <= 1:
        row = data.reshape(-1)
        return np.vstack([row, row])
    if str(envmode).lower() == "rms":
        positive = np.where(data > 0, data, 0.0)
        negative = np.where(data < 0, data, 0.0)
        pos_counts = np.maximum((data > 0).sum(axis=0), 1)
        neg_counts = np.maximum((data < 0).sum(axis=0), 1)
        return np.vstack(
            [np.sqrt(np.sum(positive**2, axis=0) / pos_counts), -np.sqrt(np.sum(negative**2, axis=0) / neg_counts)]
        )
    return np.vstack([data.max(axis=0), data.min(axis=0)])


def _normalize_sortvar(sortvar):
    mode = str(sortvar).lower()
    mode = _SORTVAR_ALIASES.get(mode, mode)
    if mode not in _VALID_SORTVAR:
        raise ValueError(f"envtopo: unknown 'sortvar' value {sortvar!r}; expected one of {_VALID_SORTVAR}")
    return mode


def _time_axis(timerange, n_frames):
    if _is_empty(timerange):
        return np.arange(n_frames, dtype=float)
    bounds = np.asarray(timerange, dtype=float).ravel()
    if bounds.size != 2:
        raise ValueError("timerange must be [min max] in milliseconds")
    return np.linspace(bounds[0], bounds[1], n_frames)


def _resolve_channels(plotchans, n_chans):
    if _is_empty(plotchans):
        return np.arange(n_chans, dtype=int)
    indices = np.asarray(plotchans, dtype=int).ravel() - 1
    if np.any(indices < 0) or np.any(indices >= n_chans):
        raise ValueError("plotchans are outside the available channels")
    return indices


def _resolve_components(compnums, n_components):
    if _is_empty(compnums):
        return np.arange(n_components, dtype=int)
    indices = np.asarray(compnums, dtype=int).ravel() - 1
    if np.any(indices < 0) or np.any(indices >= n_components):
        raise ValueError("compnums are outside the available ICA components")
    _, first = np.unique(indices, return_index=True)
    return indices[np.sort(first)]


def _resolve_subcomps(subcomps, n_components, candidates):
    if subcomps is None:
        return np.asarray([], dtype=int)
    values = np.asarray(subcomps).ravel()
    if values.size == 0:  # empty list -> remove all but the candidate components
        return np.setdiff1d(np.arange(n_components, dtype=int), candidates)
    if np.any(values < 1):  # a 0 (or negative) entry means "remove none"
        return np.asarray([], dtype=int)
    indices = values.astype(int) - 1
    if np.any(indices >= n_components):
        raise ValueError("subcomps are outside the available ICA components")
    return indices


def _limcontrib_frames(limcontrib, times_ms, n_frames):
    if _is_empty(limcontrib):
        return 0, n_frames - 1
    bounds = np.asarray(limcontrib, dtype=float).ravel()
    if bounds.size != 2 or not np.any(bounds != 0):
        return 0, n_frames - 1
    start, stop = float(times_ms[0]), float(times_ms[-1])
    per_ms = (n_frames - 1) / (stop - start)
    lim1 = int(round((min(max(bounds[0], start), stop) - start) * per_ms))
    lim2 = int(round((min(max(bounds[1], start), stop) - start) * per_ms))
    if lim2 <= lim1:
        raise ValueError("limcontrib does not span any samples")
    return lim1, lim2


def _build_figure(
    *,
    times_ms,
    data_env,
    summed_env,
    comp_envelopes,
    max_projections,
    plotted_frames,
    plotted_labels,
    plot_channels,
    chanlocs,
    limcontrib_ms,
    envmode,
    title,
    topoplot_options,
):
    """Draw the envelope panel, summed/per-component envelopes and scalp maps."""
    locs = chanlocs_as_list(chanlocs) if chanlocs is not None else []
    draw_maps = bool(locs)
    n_topos = plotted_labels.size

    figure = plt.figure(figsize=(9, 6))
    env_ax = figure.add_axes([0.10, 0.10, 0.80, 0.52] if draw_maps else [0.10, 0.12, 0.85, 0.78])

    times = np.asarray(times_ms, dtype=float) / 1000.0  # EEGLAB envtopo plots the time axis in seconds

    env_ax.fill_between(times, summed_env[1], summed_env[0], color=_FILLCOLOR, linewidth=0.0, zorder=1)
    for position in range(n_topos):
        color = _COMP_COLORS[position % len(_COMP_COLORS)]
        env_ax.plot(times, comp_envelopes[position, 0], color=color, linewidth=1.0, zorder=3)
        env_ax.plot(times, comp_envelopes[position, 1], color=color, linewidth=1.0, zorder=3)
    env_ax.plot(times, data_env[0], color="k", linewidth=2.0, zorder=4)
    env_ax.plot(times, data_env[1], color="k", linewidth=2.0, zorder=4)

    env_ax.set_xlim(float(times[0]), float(times[-1]))
    env_ax.set_xlabel("Time (s)")
    env_ax.set_ylabel("RMS (µV)" if str(envmode).lower() == "rms" else "Potential (µV)")
    # Denser y-ticks (~5 µV spacing) to match EEGLAB rather than matplotlib's sparse default.
    env_ax.yaxis.set_major_locator(MaxNLocator(nbins=11, steps=[1, 2, 5, 10], min_n_ticks=8))
    env_ax.grid(True, axis="y", linestyle=":")
    if times[0] < 0 < times[-1]:
        env_ax.axvline(0.0, color="k", linewidth=1.5, zorder=2)
    if limcontrib_ms is not None:
        for edge in limcontrib_ms:
            env_ax.axvline(float(edge) / 1000.0, color="k", linestyle=":", linewidth=1.2, zorder=2)

    if draw_maps:
        _draw_maps_row(
            figure,
            env_ax,
            times,
            comp_envelopes,
            max_projections,
            plotted_frames,
            plotted_labels,
            plot_channels,
            locs,
            topoplot_options,
        )
    if title:
        figure.suptitle(title, fontsize=12, fontweight="bold")
    return figure


def _draw_maps_row(
    figure,
    env_ax,
    times,
    comp_envelopes,
    max_projections,
    plotted_frames,
    plotted_labels,
    plot_channels,
    locs,
    topoplot_options,
):
    """Top row of scalp maps in temporal order, joined to their peak latency."""
    n_topos = plotted_labels.size
    temporal = np.argsort(plotted_frames)
    top_y, top_h = 0.68, 0.24
    left, right = 0.10, 0.88
    slot = (right - left) / n_topos
    map_w = min(slot * 0.9, 0.22)
    options = {"electrodes": "on" if map_w >= 0.12 else "off", "maplimits": "absmax", **(topoplot_options or {})}

    for column, source in enumerate(temporal):
        latency = float(times[plotted_frames[source]])
        center = left + slot * (column + 0.5)
        topo_ax = figure.add_axes([center - map_w / 2, top_y, map_w, top_h])
        topoplot(max_projections[plot_channels, source], locs, axes=topo_ax, **options)
        topo_ax.set_title(f"IC {int(plotted_labels[source])}", fontsize=10, fontweight="bold")

        color = _COMP_COLORS[source % len(_COMP_COLORS)]
        figure.add_artist(
            ConnectionPatch(
                xyA=(latency, comp_envelopes[source, 0, plotted_frames[source]]),
                coordsA=env_ax.transData,
                xyB=(0.5, 0.0),
                coordsB=topo_ax.transAxes,
                color=color,
                linewidth=1.0,
            )
        )

    cbar_ax = figure.add_axes([0.925, top_y + 0.05, 0.018, top_h - 0.10])
    cmap = plt.get_cmap((topoplot_options or {}).get("colormap") or "turbo")
    colorbar = figure.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax)
    colorbar.set_ticks([-0.8, 0, 0.8])
    colorbar.set_ticklabels(["-", "", "+"])
    colorbar.ax.tick_params(length=0)


def _is_empty(value):
    if value is None:
        return True
    return np.asarray(value).size == 0


__all__ = ["envtopo", "EnvtopoResult"]

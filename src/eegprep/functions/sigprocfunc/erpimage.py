"""ERP image plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.sigprocfunc.topoplot import plot_channel_location

_ZERO_LINEWIDTH = 2.5  # solid time-zero line, clearly thicker than dotted vert lines (EEGLAB ZEROWIDTH=3.0)


def erpimage(
    data: Any,
    *,
    times: Any = None,
    title: str = "",
    sort_values: Any = None,
    smooth: Any = None,
    decimate: int = 1,
    caxis: Any = None,
    cbar: bool = True,
    plot_erp: bool = True,
    vert: Any = None,
    chan_locs: Any = None,
    channel_index: int | None = None,
    target: Any = None,
):
    """Plot trials as an EEGLAB-style ERP image plus the average ERP.

    Pass ``chan_locs`` and ``channel_index`` (1-based) to draw a small scalp
    map above the image with the plotted channel marked, matching EEGLAB's
    default ERP-image layout for channels.

    ``caxis`` accepts either ``[lo, hi]`` explicit limits or a single fraction
    ``f`` that sets a symmetric color axis of ``+/- f * max(|image|)`` (EEGLAB
    ``erpimage.m`` ``caxfraction``). Pass ``target`` (a Figure or SubFigure) to
    draw the panels into an existing figure instead of opening a new window.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("erpimage data must be points x trials")
    points, trials = values.shape
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(points)
    )
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    order = _trial_order(values, sort_values)
    image = values[:, order].T
    image = _decimate_trials(image, decimate)
    image = _smooth_trials(image, smooth)
    show_topo = chan_locs is not None and channel_index is not None
    height_ratios: list[float] = []
    if show_topo:
        height_ratios.append(1.0)
    height_ratios.append(3.0)
    if plot_erp:
        height_ratios.append(1.0)
    fig_height = 1.2 * sum(height_ratios) + 0.6
    fig = target if target is not None else plt.figure(figsize=(7.5, fig_height))
    # Reserve a narrow colorbar column only when a colorbar is drawn, so cbar=False fills the width.
    gs = fig.add_gridspec(
        nrows=len(height_ratios),
        ncols=2 if cbar else 1,
        width_ratios=[20, 1] if cbar else [1],
        height_ratios=height_ratios,
        hspace=0.15,
        wspace=0.04,
    )
    row = 0
    topo_ax = fig.add_subplot(gs[row, 0]) if show_topo else None
    if topo_ax is not None:
        row += 1
        cell = topo_ax.get_position()
        fw, fh = fig.get_size_inches()
        side_h = cell.height
        side_w = side_h * fh / fw  # keep the scalp map square in display coordinates
        # EEGLAB draws the scalp map as a small square at the upper left (erpimage.m).
        topo_ax.set_position((cell.x0 + 0.10 * cell.width, cell.y0, side_w, side_h))
    image_ax = fig.add_subplot(gs[row, 0])
    cax = fig.add_subplot(gs[row, 1]) if cbar else None
    row += 1
    erp_ax = fig.add_subplot(gs[row, 0], sharex=image_ax) if plot_erp else None
    extent = [float(x_values[0]), float(x_values[-1]), 1, image.shape[0]]
    draw_zero = float(x_values[0]) <= 0.0 <= float(x_values[-1])
    im = image_ax.imshow(image, aspect="auto", origin="lower", extent=extent, cmap="turbo")
    limits = _limits(caxis)
    if limits is None:
        # EEGLAB erpimage default: color axis symmetric about 0, optionally scaled
        # by a single caxis fraction f -> +/- f * max(|image|) (erpimage.m).
        cmax = float(np.nanmax(np.abs(image))) if image.size else 0.0
        fraction = _caxis_fraction(caxis)
        if fraction is not None:
            cmax *= fraction
        limits = (-cmax, cmax) if np.isfinite(cmax) and cmax > 0 else None
    if limits is not None:
        im.set_clim(*limits)
    image_ax.set_ylabel("Trials")
    image_ax.set_title(title or "ERP image")
    for latency in _numeric_values(vert):
        image_ax.axvline(latency, color="black", linestyle=":", linewidth=0.8)
    if draw_zero:
        image_ax.axvline(0, color="black", linewidth=_ZERO_LINEWIDTH)
    if cax is not None:
        colorbar = fig.colorbar(im, cax=cax)
        vmin, vmax = im.get_clim()
        if vmax > vmin:
            # EEGLAB cbar: 5 ticks across the color range, decade-rounded labels (cbar.m).
            ticks, tick_labels = _colorbar_ticks(vmin, vmax)
            colorbar.set_ticks(ticks)
            colorbar.set_ticklabels([f"{value:g}" for value in tick_labels])
    if erp_ax is not None:
        erp_ax.plot(x_values, np.nanmean(values, axis=1), color="black")
        erp_ax.axhline(0, color="0.7", linewidth=0.6)
        for latency in _numeric_values(vert):
            erp_ax.axvline(latency, color="black", linestyle=":", linewidth=0.8)
        if draw_zero:
            erp_ax.axvline(0, color="black", linewidth=_ZERO_LINEWIDTH)
        erp_ax.set_xlabel("Time (ms)")
        erp_ax.set_ylabel("µV")
    else:
        image_ax.set_xlabel("Time (ms)")
    if topo_ax is not None:
        plot_channel_location(topo_ax, chan_locs, int(channel_index))
    return fig, image


def _trial_order(values: np.ndarray, sort_values: Any) -> np.ndarray:
    if sort_values is None or (isinstance(sort_values, (list, tuple)) and not sort_values):
        return np.arange(values.shape[1])
    sort_array = np.asarray(sort_values, dtype=float).ravel()
    if sort_array.size != values.shape[1]:
        raise ValueError("sort_values must contain one value per trial")
    return np.argsort(sort_array)


def _decimate_trials(image: np.ndarray, decimate: int) -> np.ndarray:
    step = max(1, int(decimate or 1))
    return image[::step, :]


def _smooth_trials(image: np.ndarray, smooth: Any) -> np.ndarray:
    values = _numeric_values(smooth)
    if values.size == 0 or values[0] <= 1:
        return image
    window = min(int(values[0]), image.shape[0])
    kernel = np.ones(window, dtype=float) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, image)


def _limits(value: Any) -> tuple[float, float] | None:
    values = _numeric_values(value)
    if values.size != 2 or np.all(values == 0):
        return None
    return float(values[0]), float(values[1])


def _caxis_fraction(value: Any) -> float | None:
    """A single ``caxis`` value is an EEGLAB caxfraction, not explicit limits."""
    values = _numeric_values(value)
    return float(values[0]) if values.size == 1 else None


def _numeric_values(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    values = np.asarray(value, dtype=float).ravel()
    return values[np.isfinite(values)]


def _colorbar_ticks(vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Five evenly spaced ticks with EEGLAB cbar's decade-based label rounding (cbar.m)."""
    ticks = np.linspace(vmin, vmax, 5)
    scale = max(abs(vmin), abs(vmax))
    dec = int(np.floor(np.log10(scale)))
    if dec < 1:
        labels = np.round(ticks * 10.0 ** (1 - dec)) * 10.0 ** (dec - 1)
    elif dec == 1:
        labels = np.round(ticks * 10.0 ** (2 - dec)) * 10.0 ** (dec - 2)
    else:
        labels = np.round(ticks)
    return ticks, labels


__all__ = ["erpimage"]

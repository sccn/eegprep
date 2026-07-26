"""Power spectra plotting helper matching EEGLAB ``spectopo`` basics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from scipy.signal import get_window, welch

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot

# Lowest frequency plotted on the spectra axis, matching EEGLAB ``spectopo`` LOPLOTHZ.
LOPLOTHZ = 1.0
# Per-channel trace colors, mirroring EEGLAB ``spectopo`` ``allcolors``. Channel
# ``k`` (1-based) uses ``_TRACE_COLORS[k % len]``, as EEGLAB does with ``mod(k, 7) + 1``.
_TRACE_COLORS = [
    (0.0, 0.75, 0.75),
    (1.0, 0.0, 0.0),
    (0.0, 0.5, 0.0),
    (0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25),
    (0.75, 0.75, 0.0),
    (0.75, 0.0, 0.75),
]


def spectopo(
    data: np.ndarray,
    frames: int,
    srate: float,
    *,
    percent: float = 100,
    freqs: Any = None,
    freqrange: Any = None,
    chanlocs: Any = None,
    topoplot_options: dict[str, Any] | None = None,
    title: str = "",
    plot: str = "on",
    winsize: int | None = None,
    overlap: int = 0,
    nfft: int | None = None,
):
    """Compute and optionally plot channel/component log power spectra.

    This implements the noninteractive EEGLAB ``spectopo`` core used by
    ``pop_spectopo``: data are channel-major, Welch spectra are reported in dB,
    and optional scalp maps are drawn at requested frequencies when channel
    locations are available.
    """
    spectra, frequency_values, specstd = compute_spectra(
        data,
        frames,
        srate,
        percent=percent,
        winsize=winsize,
        overlap=overlap,
        nfft=nfft,
    )
    figure = None
    if str(plot).lower() != "off":
        figure = plot_spectra(
            spectra,
            frequency_values,
            freqs=freqs,
            freqrange=freqrange,
            chanlocs=chanlocs,
            topoplot_options=topoplot_options,
            title=title,
        )
    return spectra, frequency_values, None, None, specstd, figure


def compute_spectra(
    data: np.ndarray,
    frames: int,
    srate: float,
    *,
    percent: float = 100,
    winsize: int | None = None,
    overlap: int = 0,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray, None]:
    """Return Welch spectra in dB as ``channels x frequencies``.

    Epoched input (3-D ``(nchan, pnts, trials)`` or 2-D ``(nchan, pnts*trials)`` with
    ``frames < sample_count``) is handled per epoch and averaged in linear power
    before the dB conversion, matching EEGLAB ``spectopo``'s ``spectcomp``. 2-D
    concatenated data is unstacked column-major to match MATLAB's ``reshape``.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        _, pnts, trials = values.shape
        epochs = values
    elif values.ndim == 2:
        nchan, sample_count = values.shape
        frames = int(frames) if frames > 0 else sample_count
        if sample_count % frames:
            raise ValueError("compute_spectra: sample count is not an integer multiple of frames")
        trials = sample_count // frames
        # column-major so trial ``k`` occupies samples ``[k*frames:(k+1)*frames]``,
        # matching MATLAB's ``reshape(data, nchan, pnts, trials)``
        epochs = values.reshape(nchan, frames, trials, order="F") if trials > 1 else values[:, :, np.newaxis]
        pnts = frames
    else:
        raise ValueError("data must be a 2-D or 3-D channel-major array")

    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in the range (0, 100]")
    if percent < 100:
        if trials > 1:
            keep_trials = max(1, int(round(trials * percent / 100.0)))
            epochs = epochs[:, :, :keep_trials]
            trials = keep_trials
        else:
            keep = max(1, int(round(pnts * percent / 100.0)))
            epochs = epochs[:, :keep, :]
            pnts = keep

    nperseg = int(winsize or min(round(srate), pnts))
    nperseg = max(1, min(nperseg, pnts))
    noverlap = max(0, min(int(overlap), nperseg - 1))
    # symmetric Hamming + no detrend to match MATLAB pwelch
    window = get_window("hamming", nperseg, fftbins=False)
    freqs = None
    psd_sum: np.ndarray | None = None
    for index in range(trials):
        freqs, power = welch(
            epochs[:, :, index],
            fs=float(srate),
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            axis=1,
            detrend=False,
            scaling="density",
        )
        if psd_sum is None:
            psd_sum = np.zeros_like(power)
        psd_sum += power
    psd_mean = psd_sum / trials
    spectra = 10.0 * np.log10(np.maximum(psd_mean, np.finfo(float).tiny))
    return spectra, freqs, None


def plot_spectra(
    spectra: np.ndarray,
    frequency_values: np.ndarray,
    *,
    freqs: Any = None,
    freqrange: Any = None,
    chanlocs: Any = None,
    topoplot_options: dict[str, Any] | None = None,
    title: str = "",
):
    """Plot spectra and optional scalp maps, mirroring EEGLAB ``spectopo``.

    Scalp maps sit in a top row above the spectra axis, connected to vertical
    frequency markers by leader lines, with a ``+``/``-`` polarity colorbar on the
    right, as in EEGLAB.
    """
    requested_freqs = np.sort(_numeric_values(freqs))
    scalp_values, scalp_labels = _scalp_maps(spectra, frequency_values, requested_freqs)
    locs = chanlocs_as_list(chanlocs)
    draw_maps = bool(scalp_values) and bool(locs)

    if draw_maps:
        fig = plt.figure(figsize=(7.6, 6.2))
        spec_ax = fig.add_axes([0.13, 0.10, 0.80, 0.52])
    else:
        fig, spec_ax = plt.subplots(figsize=(7, 4))

    for index, channel_spectrum in enumerate(spectra):
        spec_ax.plot(
            frequency_values, channel_spectrum, color=_TRACE_COLORS[(index + 1) % len(_TRACE_COLORS)], linewidth=2.0
        )
    spec_ax.set_xlabel("Frequency (Hz)")
    spec_ax.set_ylabel(r"Log Power Spectral Density 10*log$_{10}$($\mu$V$^2$/Hz)")
    spec_ax.spines[["top", "right"]].set_visible(False)

    low, high, min_idx, max_idx = _frequency_window(frequency_values, requested_freqs, freqrange)
    spec_ax.set_xlim(low, high)
    y_low, y_high = _spectra_ylim(spectra, min_idx, max_idx)
    if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
        spec_ax.set_ylim(y_low, y_high)

    if draw_maps:
        _draw_maps_row(
            fig,
            spec_ax,
            scalp_values,
            scalp_labels,
            locs,
            requested_freqs,
            frequency_values,
            spectra,
            topoplot_options,
        )

    if title:
        fig.suptitle(title, fontsize=12)
    if not draw_maps:
        fig.tight_layout()
    return fig


def _frequency_window(
    frequency_values: np.ndarray, requested_freqs: np.ndarray, freqrange: Any
) -> tuple[float, float, int, int]:
    """Return the (low, high) x-limits and their frequency indices, as EEGLAB does."""
    bounds = _numeric_values(freqrange)
    if bounds.size >= 2:
        low, high = float(bounds[0]), float(bounds[1])
    else:
        low = LOPLOTHZ
        maxfreq = float(np.nanmax(requested_freqs)) if requested_freqs.size else float(np.nanmax(frequency_values))
        high = 5.0 * np.ceil(maxfreq / 5.0) if maxfreq % 5 != 0 else maxfreq * 1.1
    min_idx = int(np.argmin(np.abs(frequency_values - low)))
    max_idx = int(np.argmin(np.abs(frequency_values - high)))
    return low, high, min_idx, max_idx


def _spectra_ylim(spectra: np.ndarray, min_idx: int, max_idx: int) -> tuple[float, float]:
    """Data range over the plotted band, expanded by 1/7 each side (EEGLAB convention)."""
    low_i, high_i = sorted((min_idx, max_idx))
    window = spectra[:, low_i : high_i + 1]
    if window.size == 0:
        return np.nan, np.nan
    y_low, y_high = float(np.nanmin(window)), float(np.nanmax(window))
    span = y_high - y_low
    return y_low - span / 7.0, y_high + span / 7.0


def _draw_maps_row(
    fig: Any,
    spec_ax: Any,
    scalp_values: list[np.ndarray],
    scalp_labels: list[str],
    locs: list,
    requested_freqs: np.ndarray | None,
    frequency_values: np.ndarray,
    spectra: np.ndarray,
    topoplot_options: dict[str, Any] | None,
) -> None:
    """Draw the top row of scalp maps, the polarity colorbar, and the vertical
    frequency markers plus leader lines to each map.

    Each map is scaled independently (``maplimits='absmax'``), so the shared
    colorbar is polarity-only (``+``/``-``), not a common data scale."""
    count = len(scalp_values)
    top_y, top_h = 0.66, 0.26
    left, right = 0.10, 0.88
    slot = (right - left) / count
    map_w = min(slot * 0.92, 0.24)
    plot_options = {"electrodes": "off", "maplimits": "absmax", **(topoplot_options or {})}

    map_axes = []
    for index, (values, label) in enumerate(zip(scalp_values, scalp_labels)):
        center = left + slot * (index + 0.5)
        topo_ax = fig.add_axes([center - map_w / 2, top_y, map_w, top_h])
        topoplot(values, locs, axes=topo_ax, **plot_options)
        topo_ax.set_title(label, fontweight="bold", fontsize=11)
        map_axes.append(topo_ax)

    cbar_ax = fig.add_axes([0.92, top_y + 0.02, 0.02, top_h - 0.06])
    cmap = plt.get_cmap((topoplot_options or {}).get("colormap") or "turbo")
    colorbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax)
    # Inset the +/- labels from the bar ends (EEGLAB places them at the outermost
    # ticks, not flush with the extremes). Purely cosmetic; the bar still spans the
    # full gradient and does not drive the per-map scalp colors.
    colorbar.set_ticks([-0.8, 0, 0.8])
    colorbar.set_ticklabels(["-", "", "+"])
    colorbar.ax.tick_params(length=0)

    for topo_ax, freq in zip(map_axes, requested_freqs):
        freq_index = int(np.argmin(np.abs(frequency_values - freq)))
        column = spectra[:, freq_index]
        y_low, y_high = float(np.nanmin(column)), float(np.nanmax(column))
        spec_ax.plot([freq, freq], [y_low, y_high], color="k", linewidth=2.0, zorder=5)
        fig.add_artist(
            ConnectionPatch(
                xyA=(freq, y_high),
                coordsA=spec_ax.transData,
                xyB=(0.5, 0.05),
                coordsB=topo_ax.transAxes,
                color="k",
                linewidth=0.5,
            )
        )


def _scalp_maps(
    spectra: np.ndarray,
    frequency_values: np.ndarray,
    requested_freqs: np.ndarray,
) -> tuple[list[np.ndarray], list[str]]:
    maps = []
    labels = []
    last = requested_freqs.size - 1
    for index, freq in enumerate(requested_freqs):
        freq_index = int(np.argmin(np.abs(frequency_values - freq)))
        # EEGLAB maps the mean-removed power across channels so the map shows
        # spatial deviation rather than the overall level.
        column = spectra[:, freq_index]
        maps.append(column - np.nanmean(column))
        labels.append(f"{freq:.1f} Hz" if index == last else f"{freq:.1f}")
    return maps, labels


def _numeric_values(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).ravel()
    if value == "":
        return np.asarray([], dtype=float)
    return np.asarray([value], dtype=float)


# EEGLAB spectopo top-contributing-component trace/leader colors (``colrs``).
_ICA_TRACE_COLORS = ["r", "b", "g", "m", "c"]


def _select_component_maps(comp_spectra, component_numbers, freq_index, nicamaps, icamaps):
    """Return row indices of the components to map: explicit ``icamaps`` or the
    ``nicamaps`` components with the most power at the marker frequency (EEGLAB)."""
    explicit = _numeric_values(icamaps).astype(int)
    if explicit.size:
        lookup = {int(number): row for row, number in enumerate(component_numbers)}
        return np.array([lookup[n] for n in explicit if n in lookup], dtype=int)
    order = np.argsort(comp_spectra[:, freq_index])[::-1]  # highest power first
    count = int(nicamaps) if nicamaps else 5
    return order[: min(count, comp_spectra.shape[0])]


def plot_component_spectra(
    comp_spectra: np.ndarray,
    frequency_values: np.ndarray,
    *,
    component_numbers: np.ndarray,
    icawinv: np.ndarray,
    chanlocs: Any,
    channel_spectra: np.ndarray,
    channel_chanlocs: Any = None,
    freq: Any,
    freqrange: Any = None,
    nicamaps: int = 5,
    icamaps: Any = None,
    topoplot_options: dict[str, Any] | None = None,
    title: str = "",
):
    """Plot component power spectra like EEGLAB ``spectopo`` component mode.

    Draws the bold black data RMS-power curve, thin traces for every component,
    colored traces for the top-``nicamaps`` components at ``freq``, a vertical
    marker at ``freq``, and a top row of scalp maps (the composite power-at-freq
    map plus each mapped component's scalp projection), joined to the marker by
    colored leader lines.
    """
    freqs_req = _numeric_values(freq)
    f0 = float(freqs_req[0]) if freqs_req.size else float(frequency_values[len(frequency_values) // 2])
    fidx = int(np.argmin(np.abs(frequency_values - f0)))
    ncomp = comp_spectra.shape[0]
    component_numbers = np.asarray(component_numbers, dtype=int)
    icawinv = np.asarray(icawinv, dtype=float)

    # Black curve: RMS power across channels (linear), back to dB (EEGLAB eegspecdBtoplot).
    chan_linear = 10.0 ** (np.asarray(channel_spectra, dtype=float) / 10.0)
    data_rms = 10.0 * np.log10(np.sqrt(np.nanmean(chan_linear**2, axis=0)))

    sel = _select_component_maps(comp_spectra, component_numbers, fidx, nicamaps, icamaps)
    comp_locs = chanlocs_as_list(chanlocs)
    composite_locs = chanlocs_as_list(channel_chanlocs) if channel_chanlocs is not None else comp_locs

    fig = plt.figure(figsize=(8.6, 6.4))
    spec_ax = fig.add_axes([0.12, 0.10, 0.78, 0.52])
    for row in range(ncomp):
        spec_ax.plot(
            frequency_values, comp_spectra[row], color=_TRACE_COLORS[(row + 1) % len(_TRACE_COLORS)], linewidth=0.75
        )
    for order, row in enumerate(sel):
        spec_ax.plot(frequency_values, comp_spectra[row], color=_ICA_TRACE_COLORS[order % 5], linewidth=1.75, zorder=4)
    spec_ax.plot(frequency_values, data_rms, color="k", linewidth=2.5, zorder=6)

    spec_ax.set_xlabel("Frequency (Hz)")
    spec_ax.set_ylabel(r"Log Power Spectral Density 10*log$_{10}$($\mu$V$^2$/Hz)")
    spec_ax.spines[["top", "right"]].set_visible(False)
    low, high, min_idx, max_idx = _frequency_window(frequency_values, freqs_req, freqrange)
    spec_ax.set_xlim(low, high)
    stacked = np.vstack([comp_spectra, data_rms[None, :]])
    y_low, y_high = _spectra_ylim(stacked, min_idx, max_idx)
    if np.isfinite(y_low) and np.isfinite(y_high) and y_high > y_low:
        spec_ax.set_ylim(y_low, y_high)

    marker_lo = float(min(np.nanmin(comp_spectra[:, fidx]), data_rms[fidx]))
    marker_hi = float(max(np.nanmax(comp_spectra[:, fidx]), data_rms[fidx]))
    spec_ax.plot([f0, f0], [marker_lo, marker_hi], color="k", linewidth=2.5, zorder=7)

    # Maps: composite power-at-freq map (mean-removed across channels), then each mapped component.
    composite = np.asarray(channel_spectra, dtype=float)[:, fidx]
    composite = composite - np.nanmean(composite)
    map_specs = [(composite, f"{f0:.1f} Hz", "k", data_rms[fidx], 2.5, composite_locs)]
    for order, row in enumerate(sel):
        number = int(component_numbers[row])
        map_specs.append(
            (
                # EEGLAB spectopo plots the squared projection icawinv(:,n).^2 (power magnitude,
                # always non-negative) for component maps, not the signed scalp map.
                icawinv[:, number - 1] ** 2,
                f"{number}",
                _ICA_TRACE_COLORS[order % 5],
                float(comp_spectra[row, fidx]),
                0.75,
                comp_locs,
            )
        )

    draw_maps = bool(comp_locs)
    if draw_maps:
        count = len(map_specs)
        top_y, top_h = 0.68, 0.24
        left, right = 0.07, 0.86
        slot = (right - left) / count
        map_w = min(slot * 0.9, 0.16)
        plot_options = {"electrodes": "off", "maplimits": "absmax", **(topoplot_options or {})}
        for index, (values, label, colr, yval, lw, map_locs) in enumerate(map_specs):
            center = left + slot * (index + 0.5)
            topo_ax = fig.add_axes([center - map_w / 2, top_y, map_w, top_h])
            topoplot(values, map_locs, axes=topo_ax, **plot_options)
            topo_ax.set_title(label, fontweight="bold", fontsize=11)
            fig.add_artist(
                ConnectionPatch(
                    xyA=(f0, yval),
                    coordsA=spec_ax.transData,
                    xyB=(0.5, 0.0),
                    coordsB=topo_ax.transAxes,
                    color=colr,
                    linewidth=lw,
                )
            )
        cbar_ax = fig.add_axes([0.90, top_y + 0.02, 0.02, top_h - 0.06])
        cmap = plt.get_cmap((topoplot_options or {}).get("colormap") or "turbo")
        colorbar = fig.colorbar(ScalarMappable(cmap=cmap, norm=Normalize(vmin=-1, vmax=1)), cax=cbar_ax)
        colorbar.set_ticks([-0.8, 0, 0.8])
        colorbar.set_ticklabels(["-", "", "+"])
        colorbar.ax.tick_params(length=0)

    if title:
        fig.suptitle(title, fontsize=12)
    return fig


__all__ = ["compute_spectra", "plot_component_spectra", "plot_spectra", "spectopo"]

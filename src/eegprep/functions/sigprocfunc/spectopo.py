"""Power spectra plotting helper matching EEGLAB ``spectopo`` basics."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.topoplot import topoplot


def spectopo(
    data: np.ndarray,
    frames: int,
    srate: float,
    *,
    percent: float = 100,
    freqs: Any = None,
    freqrange: Any = None,
    chanlocs: Any = None,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Welch spectra in dB as ``channels x frequencies``."""
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D or 3-D channel-major array")
    if frames <= 0:
        frames = values.shape[1]
    sample_count = values.shape[1]
    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in the range (0, 100]")
    if percent < 100:
        keep = max(1, int(round(sample_count * percent / 100.0)))
        values = values[:, :keep]
        sample_count = keep
    nperseg = int(winsize or min(round(srate), sample_count))
    nperseg = max(1, min(nperseg, sample_count))
    noverlap = max(0, min(int(overlap), nperseg - 1))
    freqs, power = welch(
        values,
        fs=float(srate),
        window="hamming",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        axis=1,
        detrend="constant",
        scaling="density",
    )
    spectra = 10.0 * np.log10(np.maximum(power, np.finfo(float).tiny))
    specstd = np.zeros_like(spectra)
    return spectra, freqs, specstd


def plot_spectra(
    spectra: np.ndarray,
    frequency_values: np.ndarray,
    *,
    freqs: Any = None,
    freqrange: Any = None,
    chanlocs: Any = None,
    title: str = "",
):
    """Plot spectra and optional scalp maps at selected frequencies."""
    requested_freqs = _numeric_values(freqs)
    if requested_freqs.size and chanlocs_as_list(chanlocs):
        rows = 1 + int(np.ceil(requested_freqs.size / 3))
        fig = plt.figure(figsize=(8, 2.8 + rows * 1.7))
        ax = fig.add_subplot(rows, 1, 1)
        topo_axes = [
            fig.add_subplot(rows, min(3, requested_freqs.size), index + 1 + min(3, requested_freqs.size))
            for index in range(requested_freqs.size)
        ]
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        topo_axes = []
    for channel_spectrum in spectra:
        ax.plot(frequency_values, channel_spectrum, linewidth=0.8)
    mean_spectrum = np.nanmean(spectra, axis=0)
    ax.plot(frequency_values, mean_spectrum, color="black", linewidth=2.0, label="mean")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Log Power Spectral Density 10*log10(uV^2/Hz)")
    ax.set_title(title or "Channel spectra and maps")
    if freqrange is not None and len(_numeric_values(freqrange)) == 2:
        bounds = _numeric_values(freqrange)
        ax.set_xlim(float(bounds[0]), float(bounds[1]))
    elif requested_freqs.size:
        ax.set_xlim(0, max(float(np.nanmax(requested_freqs)) * 1.15, 1.0))
    ax.grid(True, alpha=0.25)
    for topo_ax, freq in zip(topo_axes, requested_freqs):
        freq_index = int(np.argmin(np.abs(frequency_values - freq)))
        topoplot(spectra[:, freq_index], chanlocs_as_list(chanlocs), axes=topo_ax, electrodes="off")
        topo_ax.set_title(f"{freq:g} Hz")
    fig.tight_layout()
    return fig


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


__all__ = ["compute_spectra", "plot_spectra", "spectopo"]

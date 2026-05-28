"""Deterministic EEGLAB-style ``newcrossf`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.timefreqfunc.newtimef import (
    _as_epochs,
    _limits,
    _numeric_vector,
    _select_freqs,
    _select_times,
    _trial_stft,
    _winsize,
)


@dataclass(frozen=True)
class CrossFrequencyResult:
    """Computed coherence/cross-spectrum arrays."""

    coherence: np.ndarray
    phase: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    allcoher: np.ndarray
    alltf_x: np.ndarray
    alltf_y: np.ndarray
    figure: Any | None = None


def newcrossf(
    x: Any,
    y: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    **kwargs: Any,
) -> CrossFrequencyResult:
    """Compute an EEGLAB-like event-related coherence decomposition."""
    x_epochs = _as_epochs(x, int(frames))
    y_epochs = _as_epochs(y, int(frames))
    if x_epochs.shape != y_epochs.shape:
        raise ValueError("newcrossf inputs must have matching frames and trials")
    frames = x_epochs.shape[0]
    tlimits = _limits(tlimits, frames, float(srate))
    cycles_array = _numeric_vector(cycles)
    winsize = _winsize(frames, float(srate), cycles_array, kwargs.get("winsize"), kwargs.get("freqs"))
    padratio = int(_first_numeric(kwargs.get("padratio"), 2))
    nfft = max(winsize, int(2 ** np.ceil(np.log2(max(winsize, 1)))) * max(padratio, 1))
    noverlap = min(winsize - 1, max(0, int(_first_numeric(kwargs.get("overlap"), winsize // 2))))

    full_freqs, times_seconds, tf_x = _trial_stft(x_epochs, float(srate), winsize, noverlap, nfft)
    _freqs_y, _times_y, tf_y = _trial_stft(y_epochs, float(srate), winsize, noverlap, nfft)
    freqs, tf_x = _select_freqs(full_freqs, tf_x, kwargs.get("freqs"), kwargs.get("nfreqs"), kwargs.get("freqscale"))
    freq_indices = np.asarray([int(np.argmin(np.abs(full_freqs - freq))) for freq in freqs], dtype=int)
    tf_y = tf_y[freq_indices, :, :]
    full_times = tlimits[0] + times_seconds * 1000.0
    times, tf_x = _select_times(full_times, tf_x, kwargs.get("timesout"))
    time_indices = np.asarray([int(np.argmin(np.abs(full_times - time))) for time in times], dtype=int)
    tf_y = tf_y[:, time_indices, :]

    cross = tf_x * np.conjugate(tf_y)
    mode = str(kwargs.get("type", "phasecoher")).lower()
    if mode == "coher":
        denominator = np.sqrt(np.nanmean(np.abs(tf_x) ** 2, axis=2) * np.nanmean(np.abs(tf_y) ** 2, axis=2))
        coherence_complex = np.nanmean(cross, axis=2) / np.maximum(denominator, np.finfo(float).tiny)
        coherence = np.abs(coherence_complex)
    elif mode == "crossspec":
        coherence_complex = np.nanmean(cross, axis=2)
        coherence = np.abs(coherence_complex)
    elif mode == "phasecoher":
        unit = np.divide(cross, np.maximum(np.abs(cross), np.finfo(float).tiny))
        coherence_complex = np.nanmean(unit, axis=2)
        coherence = np.abs(coherence_complex)
    else:
        raise NotImplementedError("newcrossf currently supports type='phasecoher', 'coher', or 'crossspec'")
    phase = np.angle(coherence_complex)

    figure = None
    if _is_on(kwargs.get("plot", "on")):
        figure = _plot_cross_frequency(
            coherence,
            phase,
            times,
            freqs,
            title=str(kwargs.get("title", "Cross-coherence")),
            plotamp=_is_on(kwargs.get("plotamp", kwargs.get("plotersp", "on"))),
            plotphase=_is_on(kwargs.get("plotphase", "on")),
        )
    return CrossFrequencyResult(coherence, phase, times, freqs, cross, tf_x, tf_y, figure)


def _plot_cross_frequency(
    coherence: np.ndarray,
    phase: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    plotamp: bool,
    plotphase: bool,
):
    panels = int(plotamp) + int(plotphase)
    if panels == 0:
        return None
    fig, axes = plt.subplots(panels, 1, figsize=(7.5, 5.0), squeeze=False)
    row = 0
    if plotamp:
        image = axes[row, 0].imshow(
            coherence,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
            vmin=0,
            vmax=max(1.0, float(np.nanmax(coherence))),
        )
        axes[row, 0].set_title(title)
        axes[row, 0].set_ylabel("Frequency (Hz)")
        fig.colorbar(image, ax=axes[row, 0], label="Coherence")
        row += 1
    if plotphase:
        image = axes[row, 0].imshow(
            phase,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[row, 0].set_ylabel("Frequency (Hz)")
        axes[row, 0].set_xlabel("Time (ms)")
        fig.colorbar(image, ax=axes[row, 0], label="Phase (rad)")
    fig.tight_layout()
    return fig


def _first_numeric(value: Any, default: float) -> float:
    values = _numeric_vector(value)
    return float(values[0]) if values.size else float(default)


def _is_on(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "off", "no", "none"}


__all__ = ["CrossFrequencyResult", "newcrossf"]

"""Deterministic EEGLAB-style ``newcrossf`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.timefreqfunc.newtimef import compute_time_frequency


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
    freqs, times, tf_x = compute_time_frequency(x, frames, tlimits, srate, cycles, **kwargs)
    freqs_y, times_y, tf_y = compute_time_frequency(y, frames, tlimits, srate, cycles, **kwargs)
    if tf_x.shape != tf_y.shape or not np.array_equal(freqs, freqs_y) or not np.array_equal(times, times_y):
        raise ValueError("newcrossf inputs must have matching frames, trials, times, and frequencies")

    cross = tf_x * np.conjugate(tf_y)
    mode = str(kwargs.get("type", "phasecoher")).lower()
    if tf_x.shape[2] == 1 and mode != "crossspec":
        mode = "crossspec"
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


def _is_on(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "off", "no", "none"}


__all__ = ["CrossFrequencyResult", "newcrossf"]

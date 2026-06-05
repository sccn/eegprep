"""EEGLAB ``dftfilt2``-style complex wavelet filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.timefreqfunc.dftfilt3 import symmetric_hanning


def dftfilt2(
    freqs: Any,
    cycles: Any,
    srate: float,
    cycleinc: str = "linear",
    kind: str = "morlet",
) -> list[np.ndarray]:
    """Return EEGLAB ``dftfilt2`` wavelet filters."""
    freq_values = np.asarray(freqs, dtype=float).ravel()
    if freq_values.size == 0:
        raise ValueError("freqs must contain at least one value")
    cycle_values = _cycle_values(cycles, freq_values, cycleinc=cycleinc)
    if cycle_values.size != freq_values.size:
        raise ValueError("cycles must be scalar, length two, or match freqs")

    wavelets: list[np.ndarray] = []
    for freq, cycle_count in zip(freq_values, cycle_values):
        winlen = float(cycle_count) * float(srate) / float(freq)
        winlenint = int(np.floor(winlen))
        if winlenint % 2 == 1:
            winlenint += 1
        winval = np.linspace(winlenint / 2.0, -winlenint / 2.0, winlenint + 1)
        if str(kind).lower() == "sinus":
            wavelet = np.exp(2j * np.pi * float(freq) * winval / float(srate)) * symmetric_hanning(winval.size)
        else:
            t = float(freq) * winval / float(srate)
            p = 2.0 * np.pi
            sigma = float(cycle_count) / 5.0
            wavelet = np.exp(1j * t * p) / np.sqrt(2.0 * np.pi) * (
                np.exp(-(t**2) / (2.0 * sigma**2)) - np.sqrt(2.0) * np.exp(-(t**2) / (sigma**2) - p**2 * sigma**2 / 4.0)
            )
        wavelets.append(wavelet.astype(complex, copy=False))
    return wavelets


def _cycle_values(cycles: Any, freqs: np.ndarray, *, cycleinc: str) -> np.ndarray:
    values = np.asarray(cycles, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("cycles must contain at least one value")
    if values.size == 1:
        return np.full(freqs.shape, float(values[0]), dtype=float)
    if values.size == 2:
        if str(cycleinc).lower() == "log":
            if np.any(values <= 0):
                raise ValueError("log cycle interpolation requires positive cycles")
            return np.exp(np.linspace(np.log(values[0]), np.log(values[1]), freqs.size))
        return np.linspace(values[0], values[1], freqs.size)
    return values.astype(float)


__all__ = ["dftfilt2"]

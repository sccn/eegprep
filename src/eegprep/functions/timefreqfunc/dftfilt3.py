"""EEGLAB-style Morlet and Hanning wavelet filters."""

from __future__ import annotations

from typing import Any

import numpy as np


def dftfilt3(
    freqs: Any,
    cycles: Any,
    srate: float,
    *,
    cycleinc: str = "linear",
    winsize: int | None = None,
    timesupport: float = 7.0,
) -> tuple[list[np.ndarray] | np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return EEGLAB ``dftfilt3``-style complex wavelet filters.

    Args:
        freqs: Frequencies of interest in Hz.
        cycles: Scalar, two-value, or per-frequency cycle specification.
        srate: Sampling rate in Hz.
        cycleinc: ``"linear"`` or ``"log"`` interpolation for two cycle values.
        winsize: Optional fixed filter length. When omitted, Morlet filters use
            EEGLAB's seven-standard-deviation support and variable lengths.
        timesupport: Number of temporal standard deviations included.

    Returns:
        ``(wavelets, cycles, freqresol, timeresol)``.
    """
    freq_values = np.asarray(freqs, dtype=float).ravel()
    if freq_values.size == 0:
        raise ValueError("freqs must contain at least one value")
    cycle_values = _cycle_values(cycles, freq_values, cycleinc=cycleinc)
    if cycle_values.size != freq_values.size:
        raise ValueError("cycles must be scalar, length two, or match freqs")
    if winsize is not None:
        winsize = int(winsize)
        if winsize <= 0:
            raise ValueError("winsize must be positive")
        if winsize % 2 == 0:
            winsize += 1

    timeresol = np.zeros(freq_values.size, dtype=float)
    freqresol = np.zeros(freq_values.size, dtype=float)
    filters: list[np.ndarray] = []
    fixed_filters = np.zeros((freq_values.size, int(winsize or 0)), dtype=complex) if winsize else None

    for index, freq in enumerate(freq_values):
        if cycle_values[index] == 0:
            if winsize is None:
                raise ValueError("winsize is required for Hanning tapered FFT filters")
            time_values = np.arange(0, np.floor(winsize / 2) * 2 + 1, dtype=float) - np.floor(winsize / 2)
            wavelet = np.exp(2j * np.pi * freq * time_values / float(srate)) * symmetric_hanning(winsize)
        else:
            normalized_freq = float(freq) / float(srate)
            sigma_freq = normalized_freq / cycle_values[index]
            sigma_time = 1.0 / (2.0 * np.pi * sigma_freq)
            amplitude = 1.0 / np.sqrt(sigma_time * np.sqrt(np.pi))
            timeresol[index] = 2.0 * sigma_time / float(srate)
            freqresol[index] = 2.0 * sigma_freq * float(srate)
            if winsize is None:
                half_width = int(np.floor(sigma_time * float(timesupport) / 2.0))
            else:
                half_width = int(np.floor(winsize / 2))
            time_values = np.arange(0, half_width * 2 + 1, dtype=float) - half_width
            wavelet = amplitude * np.exp(-(time_values**2) / (2.0 * sigma_time**2)) * np.exp(
                2j * np.pi * normalized_freq * time_values
            )
        if fixed_filters is None:
            filters.append(wavelet.astype(complex, copy=False))
        else:
            fixed_filters[index, :] = wavelet

    return (filters if fixed_filters is None else fixed_filters), cycle_values, freqresol, timeresol


def symmetric_hanning(n: int) -> np.ndarray:
    """Return EEGLAB's symmetric Hanning window."""
    n = int(n)
    if n <= 0:
        raise ValueError("window length must be positive")
    if n % 2 == 0:
        half = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1, n // 2 + 1) / (n + 1)))
        return np.concatenate([half, half[::-1]])
    half = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(1, (n + 1) // 2 + 1) / (n + 1)))
    return np.concatenate([half, half[-2::-1]])


def _cycle_values(cycles: Any, freqs: np.ndarray, *, cycleinc: str) -> np.ndarray:
    values = np.asarray(cycles, dtype=float).ravel()
    if values.size == 0:
        values = np.asarray([0.0])
    if values.size == 1:
        return np.full(freqs.shape, float(values[0]), dtype=float)
    if values.size == 2:
        if str(cycleinc).lower() == "log":
            if np.any(values <= 0):
                raise ValueError("log cycle interpolation requires positive cycles")
            return np.exp(np.linspace(np.log(values[0]), np.log(values[1]), freqs.size))
        return np.linspace(values[0], values[1], freqs.size)
    return values.astype(float)


__all__ = ["dftfilt3", "symmetric_hanning"]

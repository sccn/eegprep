"""Legacy EEGLAB ``dftfilt`` helper."""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.ieee754 import _sincos, cos
from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.miscfunc.misc import round_mat


def _hanning_half_count(taper_length: int) -> int:
    if taper_length % 2 == 0:
        return taper_length // 2
    return (taper_length + 1) // 2


def _symmetric_hanning(cosine: np.ndarray, taper_length: int) -> np.ndarray:
    half = 0.5 * (1.0 - cosine)
    if taper_length % 2 == 0:
        return np.concatenate([half, half[::-1]])
    return np.concatenate([half, half[-2::-1]])


def dftfilt(length: int, maxfreq: float, cycle: float, oversmp: float, wavfact: float) -> np.ndarray:
    """Return legacy discrete Fourier filters."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    angles = linspace(-np.pi + 2.0 * np.pi / length, np.pi - 2.0 * np.pi / length, length)
    indices = np.arange(1.0, float(maxfreq) * length / float(cycle) + np.finfo(float).eps, 1.0 / float(oversmp))
    if indices.size == 0:
        return np.empty((length, 0), dtype=complex)
    phase = (indices[:, None] * np.float64(cycle)) * angles
    sines, cosines = _sincos(phase)
    filters = (cosines - 1j * sines).T
    tapers: list[tuple[int, int, int]] = []
    pieces: list[np.ndarray] = []
    for column in range(filters.shape[1]):
        discard = int(round_mat(float(wavfact) * length * column / (column + float(oversmp))))
        upper = int(round_mat(discard / 2.0))
        lower = discard - upper
        taper_length = length - discard
        tapers.append((upper, lower, taper_length))
        if taper_length > 0:
            count = _hanning_half_count(taper_length)
            pieces.append(2.0 * np.pi * np.arange(1, count + 1) / (taper_length + 1))
    cosines = cos(np.concatenate(pieces)) if pieces else np.empty(0, dtype=np.float64)
    offset = 0
    for column, (upper, lower, taper_length) in enumerate(tapers):
        if taper_length <= 0:
            filters[:, column] = 0
            continue
        count = _hanning_half_count(taper_length)
        window = _symmetric_hanning(cosines[offset : offset + count], taper_length)
        offset += count
        filters[:, column] *= np.concatenate([np.zeros(upper), window, np.zeros(lower)])
    return filters


__all__ = ["dftfilt"]

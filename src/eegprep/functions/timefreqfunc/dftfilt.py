"""Legacy EEGLAB ``dftfilt`` helper."""

from __future__ import annotations

import numpy as np

from eegprep.functions.miscfunc.ieee754 import cos, sin
from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.miscfunc.misc import round_mat


def _symmetric_hanning(n: int) -> np.ndarray:
    if n % 2 == 0:
        half = 0.5 * (1.0 - cos(2.0 * np.pi * np.arange(1, n // 2 + 1) / (n + 1)))
        return np.concatenate([half, half[::-1]])
    half = 0.5 * (1.0 - cos(2.0 * np.pi * np.arange(1, (n + 1) // 2 + 1) / (n + 1)))
    return np.concatenate([half, half[-2::-1]])


def dftfilt(length: int, maxfreq: float, cycle: float, oversmp: float, wavfact: float) -> np.ndarray:
    """Return legacy discrete Fourier filters."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    columns = []
    angles = linspace(-np.pi + 2.0 * np.pi / length, np.pi - 2.0 * np.pi / length, length)
    for index in np.arange(1.0, float(maxfreq) * length / float(cycle) + np.finfo(float).eps, 1.0 / float(oversmp)):
        phase = np.float64(index) * np.float64(cycle) * angles
        columns.append(cos(phase) - 1j * sin(phase))
    if not columns:
        return np.empty((length, 0), dtype=complex)
    filters = np.column_stack(columns)
    for column in range(filters.shape[1]):
        discard = int(round_mat(float(wavfact) * length * column / (column + float(oversmp))))
        upper = int(round_mat(discard / 2.0))
        lower = discard - upper
        taper_length = length - discard
        if taper_length <= 0:
            filters[:, column] = 0
            continue
        taper = np.concatenate([np.zeros(upper), _symmetric_hanning(taper_length), np.zeros(lower)])
        filters[:, column] *= taper
    return filters


__all__ = ["dftfilt"]

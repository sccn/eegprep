"""Legacy EEGLAB ``dftfilt`` helper."""

from __future__ import annotations

import numpy as np

from eegprep.functions.timefreqfunc.dftfilt3 import symmetric_hanning


def dftfilt(length: int, maxfreq: float, cycle: float, oversmp: float, wavfact: float) -> np.ndarray:
    """Return legacy discrete Fourier filters."""
    length = int(length)
    if length <= 0:
        raise ValueError("length must be positive")
    columns = []
    for index in np.arange(1.0, float(maxfreq) * length / float(cycle) + np.finfo(float).eps, 1.0 / float(oversmp)):
        phase = (
            1j * index * float(cycle) * np.linspace(-np.pi + 2.0 * np.pi / length, np.pi - 2.0 * np.pi / length, length)
        )
        columns.append(np.exp(-phase))
    if not columns:
        return np.empty((length, 0), dtype=complex)
    filters = np.column_stack(columns)
    for column in range(filters.shape[1]):
        discard = int(round(float(wavfact) * length * column / (column + float(oversmp))))
        upper = int(round(discard / 2.0))
        lower = discard - upper
        taper_length = length - discard
        if taper_length <= 0:
            filters[:, column] = 0
            continue
        taper = np.concatenate([np.zeros(upper), symmetric_hanning(taper_length), np.zeros(lower)])
        filters[:, column] *= taper
    return filters


__all__ = ["dftfilt"]

"""Inverse windowed-sinc FIR order helper."""

from __future__ import annotations

import numpy as np


_WINDOW_TYPES = ("rectangular", "hann", "hamming", "blackman", "kaiser")
_WINDOW_DF = (0.9, 3.1, 3.3, 5.5)
_WINDOW_DEV = (0.089, 0.0063, 0.0022, 0.0002)


def invfirwsord(wintype: str, fs: float, m: float, dev: float | None = None) -> tuple[float, float]:
    """Estimate transition bandwidth and ripple for a windowed-sinc FIR order."""
    if wintype is None or fs is None or m is None:
        raise ValueError("Not enough input arguments.")
    fs = float(fs)
    m = float(m)
    if fs <= 0:
        raise ValueError("Sampling frequency must be a positive real scalar.")
    if m <= 1:
        raise ValueError("Filter order must be greater than one.")
    key = str(wintype).lower()
    if key not in _WINDOW_TYPES:
        raise ValueError("Unknown window type.")
    index = _WINDOW_TYPES.index(key)
    if key == "kaiser":
        if dev is None:
            raise ValueError("Not enough input arguments.")
        dev = float(dev)
        if dev <= 0:
            raise ValueError("Passband deviation must be positive.")
        devdb = -20.0 * np.log10(dev)
        df = (devdb - 8) / (2.285 * 2 * np.pi * (m - 1))
        return float(df * fs), dev
    return float(_WINDOW_DF[index] / m * fs), float(_WINDOW_DEV[index])

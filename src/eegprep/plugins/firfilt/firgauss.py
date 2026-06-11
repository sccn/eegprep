"""Gaussian low-pass FIR helper used by FIRFilt response tooling."""

from __future__ import annotations

import numpy as np


def firgauss(fc: float, fs: float = 2.0) -> np.ndarray:
    """Return a Gaussian low-pass FIR coefficient vector."""
    fc = float(fc)
    fs = float(fs)
    if fc <= 0 or fs <= 0:
        raise ValueError("Cutoff frequency and sampling frequency must be positive.")
    sigf = fc / np.sqrt(2 * np.log(2))
    sigt = fs / (2 * np.pi * sigf)
    order = int(np.ceil(3 * sigt) * 2)
    x = np.arange(-order / 2, order / 2 + 1, dtype=float)
    return (1 / (np.sqrt(2 * np.pi) * sigt)) * np.exp(-(x**2) / (2 * sigt**2))

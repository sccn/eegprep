"""Real-cepstrum minimum-phase conversion for FIR filters."""

from __future__ import annotations

import numpy as np


def minphaserceps(b, *, upsampling_factor: float = 1e3, clip_threshold: float = 1e-8) -> np.ndarray:
    """Convert FIR coefficients to a minimum-phase impulse response."""
    coeffs = np.asarray(b, dtype=float).ravel()
    if coeffs.size == 0:
        raise ValueError("Filter coefficients are required.")
    n = coeffs.size
    n_fft = 2 ** int(np.ceil(np.log2(n * float(upsampling_factor))))
    half = n_fft // 2
    spectrum = np.abs(np.fft.fft(coeffs, n_fft))
    spectrum[spectrum < clip_threshold] = clip_threshold
    cepstrum = np.real(np.fft.ifft(np.log(spectrum)))
    folded = np.r_[
        cepstrum[0],
        np.r_[cepstrum[1:half], 0] + np.conj(cepstrum[: half - 1 : -1]),
        np.zeros(half - 1),
    ]
    return np.real(np.fft.ifft(np.exp(np.fft.fft(folded))))[:n]

"""Symmetric FIR window functions matching EEGLAB firfilt."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.special import i0

from eegprep.functions.miscfunc.parity import round_mat


def windows(t: str, m: float, a: float | None = None) -> np.ndarray:
    """Return a symmetric window vector."""
    if t is None or m is None:
        raise ValueError("Not enough input arguments.")
    rounded = int(round_mat(float(m)))
    if rounded != float(m):
        warnings.warn("Non-integer window length. Rounding to integer.", RuntimeWarning, stacklevel=2)
    if rounded < 1:
        raise ValueError("Invalid window length.")
    if rounded == 1:
        return np.ones(1, dtype=float)

    odd = rounded % 2 == 1
    if odd:
        x = np.arange(0, (rounded - 1) / 2 + 1, dtype=float) / (rounded - 1)
    else:
        x = np.arange(0, rounded / 2, dtype=float) / (rounded - 1)

    key = str(t).lower()
    if key == "rectangular":
        first = np.ones(x.size, dtype=float)
    elif key == "bartlett":
        first = 2 * x
    elif key == "hann":
        first = 0.5 - 0.5 * np.cos(2 * np.pi * x)
    elif key == "hamming":
        first = 0.54 - 0.46 * np.cos(2 * np.pi * x)
    elif key == "blackman":
        first = _generalized_cosine(x, (0.42, 0.5, 0.08, 0.0))
    elif key == "blackmanharris":
        first = _generalized_cosine(x, (0.35875, 0.48829, 0.14128, 0.01168))
    elif key == "kaiser":
        beta = 0.5 if a is None else float(a)
        first = i0(beta * np.sqrt(1 - (2 * x - 1) ** 2)) / i0(beta)
    elif key == "tukey":
        first = _tukey_half(x, rounded, 0.5 if a is None else float(a))
    else:
        raise ValueError("Unknown window type.")

    if odd:
        return np.r_[first, first[-2::-1]].astype(float)
    return np.r_[first, first[::-1]].astype(float)


def _generalized_cosine(x: np.ndarray, coeffs: tuple[float, float, float, float]) -> np.ndarray:
    return (
        coeffs[0]
        - coeffs[1] * np.cos(2 * np.pi * x)
        + coeffs[2] * np.cos(4 * np.pi * x)
        - coeffs[3] * np.cos(6 * np.pi * x)
    )


def _tukey_half(x: np.ndarray, m: int, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return np.ones(x.size, dtype=float)
    if alpha >= 1:
        return 0.5 - 0.5 * np.cos(2 * np.pi * x)
    m_taper = int(np.floor((m - 1) * alpha / 2) + 1)
    x_taper = 2 * np.arange(m_taper, dtype=float) / (alpha * (m - 1)) - 1
    taper = 0.5 * (1 + np.cos(np.pi * x_taper))
    return np.r_[taper, np.ones(x.size - m_taper, dtype=float)]

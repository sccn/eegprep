"""EEGLAB-compatible sphering (whitening) matrix (spher)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import sqrtm


def spher(data: Any) -> np.ndarray:
    """Return the sphering (whitening) matrix for the given data.

    Ports EEGLAB's spher.m: ``2 * inv(sqrtm(cov(data.T)))``. The result is a
    symmetric matrix ``S`` that whitens the channel covariance ``C = cov(data.T)``
    so that ``S @ C @ S.T == 4 * I``.

    Args:
        data: 2-D array shaped ``(channels, samples)``.

    Returns:
        The ``(channels, channels)`` sphering matrix.
    """
    array = np.asarray(data, dtype=float)
    # np.cov (rowvar=True, ddof=1) matches MATLAB cov(data'); atleast_2d keeps
    # the single-channel case as a 1x1 matrix.
    covariance = np.atleast_2d(np.cov(array))
    sphere = 2.0 * np.linalg.inv(sqrtm(covariance))
    # sqrtm returns a complex dtype with ~0 imaginary part for real PSD input;
    # MATLAB's result is real, so drop the negligible imaginary component.
    return np.real_if_close(sphere)


__all__ = ["spher"]

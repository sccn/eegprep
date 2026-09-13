"""One-dimensional Gaussian windows."""

from __future__ import annotations

import numpy as np


def gauss(frames: int, standard_deviations: float) -> np.ndarray:
    """Return an EEGLAB-compatible Gaussian window with a unit peak."""
    frame_count = int(frames)
    if frame_count < 1 or standard_deviations <= 0:
        raise ValueError("frames and standard_deviations must be positive")
    if frame_count == 1:
        return np.ones(1)
    locations = np.linspace(-standard_deviations, standard_deviations, frame_count)
    return np.exp(-(locations**2))


__all__ = ["gauss"]

"""One-dimensional Gaussian windows."""

from __future__ import annotations

import numpy as np

from ._validation import integer_scalar


def gauss(frames: int, standard_deviations: float) -> np.ndarray:
    """Return a Gaussian window sampled from ``-sds`` through ``sds``.

    One frame is the sample at ``-sds``, ``exp(-sds**2)``. Longer windows
    include both endpoints, so an odd frame count peaks at 1.
    """
    frame_count = integer_scalar(frames, "frames")
    if frame_count < 1 or standard_deviations <= 0:
        raise ValueError("frames and standard_deviations must be positive")
    if frame_count == 1:
        # gauss.m: incr is Inf, and -sds:Inf:sds is the single sample -sds.
        return np.array([np.exp(-(standard_deviations**2))])
    locations = np.linspace(-standard_deviations, standard_deviations, frame_count)
    return np.exp(-(locations**2))


__all__ = ["gauss"]

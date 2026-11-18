"""One-dimensional Gaussian windows."""

from __future__ import annotations

import numpy as np

from ._validation import integer_scalar
from .ieee754 import exp
from .matlab_colon import matlab_colon


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
        return np.reshape(exp(-(np.float64(standard_deviations) ** 2)), 1)
    step = 2.0 * standard_deviations / (frame_count - 1)
    locations = matlab_colon(-standard_deviations, step, standard_deviations)
    return exp(-(locations * locations))


__all__ = ["gauss"]

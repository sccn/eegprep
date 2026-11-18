"""MATLAB colon operator."""

from __future__ import annotations

import math

import numpy as np

from .misc import round_mat

_EPS = float(np.finfo(np.float64).eps)


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """Return ``start:step:stop`` using MATLAB's colon operator.

    Integer steps take their length from ``floor``. Any other step rounds the
    interval count half away from zero, then drops one interval when that step
    lands more than ``2*eps*max(|start|, |stop|)`` past ``stop``. The second
    half of the vector is reflected from the endpoint, which is what makes
    ``round(start:step:stop)`` match MATLAB.

    Args:
        start: First value.
        step: Increment. A zero step returns an empty vector.
        stop: Endpoint. Included when a step lands on it within the tolerance.

    Returns:
        Colon values, or an empty float array when the range is empty.
    """
    start = float(start)
    step = float(step)
    stop = float(stop)
    if step == 0.0 or (step > 0.0 and start > stop) or (step < 0.0 and start < stop):
        return np.array([], dtype=np.float64)

    tolerance = 2.0 * _EPS * max(abs(start), abs(stop))
    direction = math.copysign(1.0, step)
    if start == math.floor(start) and step == 1.0:
        intervals = math.floor(stop) - start
    elif start == math.floor(start) and step == math.floor(step):
        quotient = math.floor(start / step)
        remainder = start - quotient * step
        intervals = math.floor((stop - remainder) / step) - quotient
    else:
        intervals = float(round_mat((stop - start) / step))
        if direction * (start + intervals * step - stop) > tolerance:
            intervals -= 1.0

    count = int(intervals)
    if count < 0:
        return np.array([], dtype=np.float64)

    endpoint = start + count * step
    if direction * (endpoint - stop) > -tolerance:
        endpoint = stop
    values = np.empty(count + 1, dtype=np.float64)
    left = np.arange(count // 2 + 1, dtype=np.float64)
    left_index = left.astype(np.int64)
    values[left_index] = start + left * step
    values[count - left_index] = endpoint - left * step
    if count % 2 == 0:
        values[count // 2] = (start + endpoint) / 2.0
    return values

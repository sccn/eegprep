"""Envelope extraction for multichannel signals."""

from __future__ import annotations

from typing import Any

import numpy as np


def env(data: Any, timelimits: Any = None, timearray: Any = None) -> np.ndarray:
    """Return the sample-wise maximum and minimum across channels.

    When ``timearray`` is supplied, both envelope edges are interpolated from
    evenly spaced samples spanning ``timelimits``.
    """
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("data must have shape (channels, timepoints)")
    if array.shape[1] == 0:
        return np.empty((2, 0), dtype=np.result_type(array.dtype, float))
    upper = np.max(array, axis=0)
    lower = np.min(array, axis=0)
    if timearray is None:
        return np.vstack([upper, lower])
    limits = np.asarray(timelimits, dtype=float).reshape(-1)
    if limits.size != 2:
        raise ValueError("timelimits must contain start and end times")
    if upper.size > 1 and limits[0] == limits[1]:
        raise ValueError("timelimits must span a non-zero interval")
    times = np.asarray(timearray, dtype=float)
    if times.ndim > 2 or (times.ndim == 2 and min(times.shape) > 1):
        raise ValueError("timearray must be a vector")
    target = times.reshape(-1)
    source = np.linspace(limits[0], limits[1], upper.size)
    if upper.size == 1:
        interpolated_upper = np.full(target.shape, upper[0], dtype=float)
        interpolated_lower = np.full(target.shape, lower[0], dtype=float)
    else:
        interpolated_upper = _v4_interpolate(source, upper, target)
        interpolated_lower = _v4_interpolate(source, lower, target)
    envelope_upper = np.maximum(interpolated_upper, interpolated_lower)
    envelope_lower = np.minimum(interpolated_upper, interpolated_lower)
    return np.vstack([envelope_upper, envelope_lower])


def _v4_interpolate(source: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Apply the biharmonic Green's function used by MATLAB griddata v4."""
    source_distance = np.abs(source[:, None] - source[None, :])
    system = _green_function(source_distance)
    try:
        weights = np.linalg.solve(system, values)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(system, values, rcond=None)[0]
    target_distance = np.abs(target[:, None] - source[None, :])
    return _green_function(target_distance) @ weights


def _green_function(distance: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = distance**2 * (np.log(distance) - 1.0)
    return np.where(np.isfinite(values), values, 0.0)


__all__ = ["env"]

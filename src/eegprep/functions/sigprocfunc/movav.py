"""Moving averages on regularly or irregularly sampled data."""

from __future__ import annotations

from typing import Any

import numpy as np


NEAR_ZERO = 1e-22


def movav(
    data: Any,
    xvals: Any = None,
    xwidth: float | None = None,
    xadv: float | None = None,
    firstx: float | None = None,
    lastx: float | None = None,
    xwin: Any = None,
    nonorm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return moving-window averages and their x-coordinate midpoints.

    ``xvals`` may be irregular. Empty windows repeat the preceding output (or
    emit zero for the first window), matching EEGLAB's useful regularization
    behavior. A non-scalar ``xwin`` applies windowed rather than rectangular
    averaging.
    """
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("data must have shape (channels, frames)")
    if array.shape[1] == 1 and array.shape[0] > 1:
        array = array.T
    channels, frames = array.shape
    if frames < 4:
        raise ValueError("data are too short")

    coordinates = _coordinates(xvals, frames)
    first = float(np.min(coordinates) if firstx is None else firstx)
    last = float(np.max(coordinates) if lastx is None else lastx)
    advance = 1.0 if xadv is None or float(xadv) == 0 else float(xadv)
    width = (last - first) / 4.0 if xwidth is None or float(xwidth) == 0 else float(xwidth)
    if advance <= 0 or width < 0 or last < first:
        raise ValueError("xwidth, xadv, firstx, and lastx do not define increasing windows")
    window = _window(xwin)

    output_frames = int(np.floor(((last - first + advance + 1.0) - width) / advance))
    output_frames = max(output_frames, 1)
    output = np.zeros((channels, output_frames), dtype=np.result_type(array.dtype, float))
    output_x = first + width / 2.0 + np.arange(output_frames, dtype=float) * advance

    low = first
    for frame in range(output_frames):
        high = low + width
        indices = np.flatnonzero((coordinates >= low) & (coordinates < high))
        if indices.size == 0:
            if frame:
                output[:, frame] = output[:, frame - 1]
        elif window is None:
            output[:, frame] = _nanmean(array[:, indices], axis=1)
            if bool(nonorm):
                output[:, frame] *= indices.size
        else:
            window_advance = (high - low) / window.size
            if window_advance <= 0:
                raise ValueError("a non-rectangular xwin requires a positive xwidth")
            buckets = np.floor((coordinates[indices] - low) / window_advance).astype(int)
            buckets = np.clip(buckets, 0, window.size - 1)
            selected_weights = window[buckets]
            weighted = _nansum(array[:, indices] * selected_weights, axis=1)
            weight_sum = float(np.sum(selected_weights))
            output[:, frame] = weighted / weight_sum if abs(weight_sum) > NEAR_ZERO and not nonorm else weighted
        low += advance
    return output, output_x


def _coordinates(xvals: Any, frames: int) -> np.ndarray:
    if xvals is None:
        return np.arange(1, frames + 1, dtype=float)
    values = np.asarray(xvals)
    if values.size == 0 or (values.size == 1 and float(values.reshape(-1)[0]) == 0):
        return np.arange(1, frames + 1, dtype=float)
    if values.ndim > 2 or (values.ndim == 2 and min(values.shape) > 1):
        raise ValueError("xvals must be a vector")
    coordinates = np.asarray(values, dtype=float).reshape(-1)
    if coordinates.size != frames:
        raise ValueError("lengths of xvals and data must be equal")
    if np.any(np.diff(coordinates) < 0):
        raise ValueError("xvals must be increasing")
    return coordinates


def _window(xwin: Any) -> np.ndarray | None:
    if xwin is None:
        return None
    values = np.asarray(xwin)
    if values.size == 0:
        return None
    if values.ndim > 2 or (values.ndim == 2 and min(values.shape) > 1):
        raise ValueError("xwin cannot be a matrix")
    window = np.asarray(values, dtype=float).reshape(-1)
    if window.size == 1:
        if window[0] in {0, 1}:
            return None
        raise ValueError("xwin must be a vector, zero, or one")
    return window


def _nanmean(values: np.ndarray, axis: int) -> np.ndarray:
    finite = ~np.isnan(values)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, values, 0), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _nansum(values: np.ndarray, axis: int) -> np.ndarray:
    finite = ~np.isnan(values)
    total = np.sum(np.where(finite, values, 0), axis=axis)
    return np.where(np.any(finite, axis=axis), total, np.nan)


__all__ = ["movav"]

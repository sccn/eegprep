"""EEGLAB-style angular time warping."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.timefreqfunc.timewarp import _frame_vector, _mapped_positions


def angtimewarp(ev_latency: Any, new_latency: Any, angdata: Any) -> np.ndarray:
    """Warp an angular time series and wrap results to ``(-pi, pi]``."""
    ev_frames = _frame_vector(ev_latency, "evLatency")
    new_frames = _frame_vector(new_latency, "newLatency")
    if ev_frames.size != new_frames.size:
        raise ValueError("evLatency and newLatency must have the same length")
    if ev_frames.size < 2:
        raise ValueError('There should be at least two events in evlatency and newlatency, e.g. "begin" and "end"')
    if np.any(np.diff(ev_frames) < 0):
        raise ValueError("evLatency should be in ascending order")
    if np.any(np.diff(new_frames) < 0):
        raise ValueError("newLatency should be in ascending order")
    if ev_frames[0] != 1:
        ev_frames = np.sort(np.concatenate([ev_frames, np.asarray([1], dtype=int)]))
        new_frames = np.sort(np.concatenate([new_frames, np.asarray([1], dtype=int)]))
    if np.any(np.diff(ev_frames) == 0) or np.any(np.diff(new_frames) == 0):
        raise ValueError("timewarp event latencies must be unique after adding the synchronized first frame")

    angles = np.asarray(angdata, dtype=float).ravel()
    old_count = int(np.max(ev_frames))
    if angles.size < old_count:
        raise ValueError("angdata must contain at least max(evLatency) samples")
    first_new = int(np.min(new_frames))
    last_new = int(np.max(new_frames))
    mapped_positions = _mapped_positions(ev_frames, new_frames, old_count)
    shifted_positions = mapped_positions - first_new + 1.0
    warped = np.zeros(last_new - first_new + 1, dtype=float)

    old_index = 0
    for new_position in range(1, warped.size + 1):
        while old_index + 1 < shifted_positions.size and new_position > shifted_positions[old_index]:
            old_index += 1
        if old_index == 0:
            warped[0] = angles[0]
            continue
        distance = shifted_positions[old_index] - shifted_positions[old_index - 1]
        if distance == 0:
            raise ValueError("timewarp cannot interpolate across repeated output positions")
        warped[new_position - 1] = angles[old_index - 1] * (
            1.0 - (new_position - shifted_positions[old_index - 1]) / distance
        ) + angles[old_index] * (1.0 - (shifted_positions[old_index] - new_position) / distance)
    return _wrap_to_pi(warped)


def _wrap_to_pi(angles: np.ndarray, center: float = 0.0) -> np.ndarray:
    wrapped = np.mod(angles, 2.0 * np.pi)
    high = wrapped > np.pi - center
    wrapped[high] -= 2.0 * np.pi
    low = wrapped < center - np.pi
    wrapped[low] += 2.0 * np.pi
    return wrapped


__all__ = ["angtimewarp"]

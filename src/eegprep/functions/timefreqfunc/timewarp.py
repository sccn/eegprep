"""EEGLAB-style linear time-warp matrix helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def timewarp(ev_latency: Any, new_latency: Any) -> np.ndarray:
    """Return the linear interpolation matrix that warps event latencies.

    ``ev_latency`` and ``new_latency`` use EEGLAB's 1-based frame indexing.
    Multiplying the returned matrix by a column time series maps samples so the
    original event frames align with the requested event frames.
    """
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

    old_count = int(np.max(ev_frames))
    first_new = int(np.min(new_frames))
    last_new = int(np.max(new_frames))
    mapped_positions = _mapped_positions(ev_frames, new_frames, old_count)
    shifted_positions = mapped_positions - first_new + 1.0
    output = np.zeros((last_new - first_new + 1, old_count), dtype=float)

    old_index = 0
    for new_position in range(1, output.shape[0] + 1):
        while old_index + 1 < shifted_positions.size and new_position > shifted_positions[old_index]:
            old_index += 1
        if old_index == 0:
            output[0, 0] = 1.0
            continue
        distance = shifted_positions[old_index] - shifted_positions[old_index - 1]
        if distance == 0:
            raise ValueError("timewarp cannot interpolate across repeated output positions")
        output[new_position - 1, old_index - 1] = 1.0 - (new_position - shifted_positions[old_index - 1]) / distance
        output[new_position - 1, old_index] = 1.0 - (shifted_positions[old_index] - new_position) / distance
    return output


def _mapped_positions(ev_frames: np.ndarray, new_frames: np.ndarray, old_count: int) -> np.ndarray:
    positions = np.zeros(old_count, dtype=float)
    time = np.arange(1, old_count + 1, dtype=float)
    for index in range(ev_frames.size - 1):
        start = int(ev_frames[index])
        stop = int(ev_frames[index + 1])
        scale = (new_frames[index + 1] - new_frames[index]) / (ev_frames[index + 1] - ev_frames[index])
        frame_indices = np.arange(start, stop, dtype=int)
        positions[frame_indices - 1] = (time[frame_indices - 1] - ev_frames[index]) * scale + new_frames[index]
    positions[old_count - 1] = float(np.max(new_frames))
    return positions


def _frame_vector(value: Any, name: str) -> np.ndarray:
    frames = np.asarray(value, dtype=float).ravel()
    if frames.size == 0:
        raise ValueError(f"{name} must contain at least two event latencies")
    if not np.all(np.isfinite(frames)):
        raise ValueError(f"{name} must contain finite frame latencies")
    rounded = np.rint(frames)
    if not np.allclose(frames, rounded):
        raise ValueError(f"{name} must contain integer frame latencies")
    if np.any(rounded < 1):
        raise ValueError(f"{name} must use EEGLAB-style 1-based frame latencies")
    return rounded.astype(int)


__all__ = ["timewarp"]

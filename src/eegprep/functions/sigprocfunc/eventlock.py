"""Align existing epochs to per-trial event values."""

from __future__ import annotations

from typing import Any

import numpy as np


def eventlock(
    data: Any,
    frames_or_xvals: Any,
    eventvals: Any,
    medval: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Shift epochs so their event frames or time values align.

    A scalar second argument selects multi-channel mode, where ``data`` has
    shape ``(channels, frames * trials)``. Zero infers frames per trial from
    ``eventvals``. A vector selects single-channel mode, where data has shape
    ``(frames, trials)``; ``[start_ms, frames, srate_hz]`` is accepted as a
    compact time-axis specification. Positive returned shifts move data right.
    """
    array = np.asarray(data)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("eventlock: data must be a non-empty 2D array")
    event_values = np.asarray(eventvals, dtype=float).reshape(-1)
    if event_values.size == 0 or not np.isfinite(event_values).all():
        raise ValueError("eventlock: eventvals must contain one finite value per trial")

    argument = np.asarray(frames_or_xvals, dtype=float).reshape(-1)
    if argument.size == 1:
        return _eventlock_multichannel(array, argument[0], event_values, medval)
    return _eventlock_single_channel(array, argument, event_values, medval)


def _eventlock_multichannel(
    data: np.ndarray,
    frames_value: float,
    event_values: np.ndarray,
    medval: float | None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if frames_value == 0:
        if data.shape[1] % event_values.size:
            raise ValueError("eventlock: data length must be divisible by the number of event values")
        frames = data.shape[1] // event_values.size
    else:
        frames = _positive_integer(frames_value, "frames")
    if data.shape[1] % frames:
        raise ValueError("eventlock: data length must be a multiple of frames")
    trials = data.shape[1] // frames
    if event_values.size != trials:
        raise ValueError("eventlock: eventvals must contain one value per trial")

    xvals = np.arange(1, frames + 1, dtype=float)
    aligned_value, shifts = _alignment(xvals, event_values, medval)
    output = np.full(data.shape, np.nan, dtype=np.result_type(data.dtype, float))
    for trial, shift in enumerate(shifts):
        start = trial * frames
        stop = start + frames
        output[:, start:stop] = _shift(data[:, start:stop], int(shift))
    return output, aligned_value, shifts


def _eventlock_single_channel(
    data: np.ndarray,
    argument: np.ndarray,
    event_values: np.ndarray,
    medval: float | None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if argument.size == 3:
        frames = _positive_integer(argument[1], "compact x-axis frame count")
        sampling_rate = float(argument[2])
        if sampling_rate <= 0:
            raise ValueError("eventlock: compact x-axis sampling rate must be positive")
        xvals = float(argument[0]) + np.arange(frames) * 1000 / sampling_rate
    else:
        xvals = argument
        frames = xvals.size
    if frames != data.shape[0]:
        raise ValueError("eventlock: data rows must equal the number of x-axis values")
    if event_values.size != data.shape[1]:
        raise ValueError("eventlock: eventvals must contain one value per trial")
    if not np.isfinite(xvals).all():
        raise ValueError("eventlock: x-axis values must be finite")

    aligned_value, shifts = _alignment(xvals, event_values, medval)
    output = np.full(data.shape, np.nan, dtype=np.result_type(data.dtype, float))
    for trial, shift in enumerate(shifts):
        output[:, trial] = _shift(data[:, trial], int(shift))
    return output, aligned_value, shifts


def _alignment(xvals: np.ndarray, event_values: np.ndarray, medval: float | None) -> tuple[float, np.ndarray]:
    requested = float(np.median(event_values) if medval is None else medval)
    median_index = int(np.argmin(np.abs(xvals - requested)))
    event_indices = np.asarray([np.argmin(np.abs(xvals - value)) for value in event_values], dtype=int)
    return float(xvals[median_index]), median_index - event_indices


def _shift(values: np.ndarray, shift: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.result_type(values.dtype, float))
    length = values.shape[-1] if values.ndim == 2 else values.shape[0]
    if abs(shift) >= length:
        return output
    if shift > 0:
        output[..., shift:] = values[..., :-shift]
    elif shift < 0:
        output[..., :shift] = values[..., -shift:]
    else:
        output[...] = values
    return output


def _positive_integer(value: Any, name: str) -> int:
    result = int(value)
    if result != value or result < 1:
        raise ValueError(f"eventlock: {name} must be a positive integer")
    return result


__all__ = ["eventlock"]

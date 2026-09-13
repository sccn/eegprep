"""Absolute peak extraction for channel-major data."""

from __future__ import annotations

from typing import Any

import numpy as np


def abspeak(data: Any, frames_per_epoch: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute peaks, zero-based frames, and signs for each epoch.

    NaNs are ignored when finite or infinite observations exist. An all-NaN
    channel/epoch returns ``NaN`` amplitude, frame ``-1``, and sign ``NaN``.
    Tied peaks use the last frame, matching EEGLAB's stable-sort behavior.
    """
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D channel-by-frame array")
    total_frames = values.shape[1]
    epoch_frames = total_frames if frames_per_epoch in {None, 0} else int(frames_per_epoch)
    if epoch_frames <= 0 or total_frames % epoch_frames:
        raise ValueError("frames_per_epoch must be positive and divide the data length")

    epochs = total_frames // epoch_frames
    reshaped = values.reshape(values.shape[0], epochs, epoch_frames)
    amplitudes = np.empty((values.shape[0], epochs), dtype=float)
    frames = np.full((values.shape[0], epochs), -1, dtype=int)
    signs = np.empty((values.shape[0], epochs), dtype=float)
    for channel in range(values.shape[0]):
        for epoch in range(epochs):
            row = reshaped[channel, epoch]
            valid = ~np.isnan(row)
            if not valid.any():
                amplitudes[channel, epoch] = np.nan
                signs[channel, epoch] = np.nan
                continue
            peak = np.nanmax(np.abs(row))
            frame = int(np.flatnonzero(valid & (np.abs(row) == peak))[-1])
            amplitudes[channel, epoch] = peak
            frames[channel, epoch] = frame
            signs[channel, epoch] = np.sign(row[frame])
    return amplitudes, frames, signs


__all__ = ["abspeak"]

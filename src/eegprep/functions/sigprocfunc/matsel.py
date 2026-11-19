"""Selection from flattened channel-by-frame epoch matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc._validation import integer_array, integer_scalar


def matsel(
    data: Any,
    frames_per_epoch: int | None,
    frames: Any | None,
    channels: Any | None = None,
    epochs: Any | None = None,
) -> np.ndarray:
    """Select 1-based channels, within-epoch frames, and epochs.

    A scalar ``0`` selects every index of that argument. An empty frame list
    selects no frames; an empty channel or epoch list selects every one.
    Input data remain two-dimensional with epochs concatenated along columns.
    """
    values = np.asarray(data)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("data must be a nonempty 2-D channel-by-frame matrix")
    epoch_frames = values.shape[1] if frames_per_epoch is None else integer_scalar(frames_per_epoch, "frames_per_epoch")
    if epoch_frames == 0:
        epoch_frames = values.shape[1]
    if epoch_frames <= 0 or values.shape[1] % epoch_frames:
        raise ValueError("frames_per_epoch must divide the data length")
    epoch_count = values.shape[1] // epoch_frames
    frame_indices = _indices(frames, epoch_frames, "frames", empty_means_all=False)
    channel_indices = _indices(channels, values.shape[0], "channels", empty_means_all=True)
    epoch_indices = _indices(epochs, epoch_count, "epochs", empty_means_all=True)
    if frame_indices.size == 0 or epoch_indices.size == 0:
        return values[np.ix_(channel_indices, np.asarray([], dtype=int))]
    sample_indices = np.concatenate([frame_indices + epoch * epoch_frames for epoch in epoch_indices])
    return values[np.ix_(channel_indices, sample_indices)]


def _indices(selection: Any | None, length: int, name: str, *, empty_means_all: bool) -> np.ndarray:
    if selection is None:
        return np.arange(length)
    array = np.asarray(selection)
    if array.size == 0:
        return np.arange(length) if empty_means_all else np.asarray([], dtype=int)
    if np.all(array == 0):
        return np.arange(length)
    indices = integer_array(selection, name).reshape(-1) - 1
    if np.any(indices < 0) or np.any(indices >= length):
        raise IndexError(f"{name} indices are out of range")
    return indices


__all__ = ["matsel"]

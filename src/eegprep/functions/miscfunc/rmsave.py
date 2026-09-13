"""Root-mean-square summaries for channel-major EEG data."""

from __future__ import annotations

from typing import Any

import numpy as np


def rmsave(data: Any, frames: int | None = None) -> np.ndarray:
    """Return the RMS of each channel in consecutive equal-length blocks.

    A three-dimensional ``(channels, points, trials)`` input is flattened in
    MATLAB column order. With no explicit ``frames``, each trial is one block.
    """
    array = np.asarray(data)
    if array.ndim not in {2, 3} or array.shape[0] == 0:
        raise ValueError("rmsave: data must be a non-empty 2D or 3D channel-major array")

    default_frames = array.shape[1]
    flattened = array.reshape(array.shape[0], -1, order="F").astype(np.result_type(array.dtype, float), copy=False)
    block_length = default_frames if frames is None else _positive_integer(frames)
    if flattened.shape[1] % block_length:
        raise ValueError("rmsave: frames must divide the data length exactly")

    blocks = flattened.reshape(array.shape[0], block_length, -1, order="F")
    return np.sqrt(np.mean(np.abs(blocks) ** 2, axis=1))


def _positive_integer(value: Any) -> int:
    result = int(value)
    if result != value or result < 1:
        raise ValueError("rmsave: frames must be a positive integer")
    return result


__all__ = ["rmsave"]

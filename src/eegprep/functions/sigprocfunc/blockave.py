"""Average concatenated, equal-length data epochs."""

from __future__ import annotations

from typing import Any

import numpy as np


def blockave(data: Any, frames: int, epochs: Any = None, weights: Any = None) -> np.ndarray:
    """Return an epoch average from channel-major concatenated data.

    Args:
        data: Array shaped ``(channels, frames * epochs)``. A one-dimensional
            input is treated as one channel.
        frames: Number of samples in each epoch.
        epochs: Optional EEGLAB-facing 1-based epoch indices. Empty or zero
            selects every epoch.
        weights: Optional weight for every epoch. The selected weights are
            normalized to sum to one.

    Returns:
        The average shaped ``(channels, frames)``.
    """
    array = np.asarray(data)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("data must have shape (channels, frames * epochs)")
    if not isinstance(frames, (int, np.integer)) or int(frames) <= 0:
        raise ValueError("frames must be a positive integer")
    frames = int(frames)
    if array.shape[1] % frames:
        raise ValueError("frames must divide the data length")

    epoch_count = array.shape[1] // frames
    selected = _selected_epochs(epochs, epoch_count)
    blocks = array.reshape(array.shape[0], epoch_count, frames)[:, selected, :]
    selected_weights = _selected_weights(weights, selected, epoch_count)
    return np.average(blocks, axis=1, weights=selected_weights)


def _selected_epochs(epochs: Any, epoch_count: int) -> np.ndarray:
    if epochs is None:
        return np.arange(epoch_count, dtype=int)
    values = np.asarray(epochs)
    if values.size == 0 or (values.size == 1 and float(values.reshape(-1)[0]) == 0):
        return np.arange(epoch_count, dtype=int)
    numeric = np.asarray(values, dtype=float).reshape(-1)
    if np.any(numeric != np.floor(numeric)):
        raise ValueError("epoch indices must be integers")
    selected = numeric.astype(int) - 1
    if np.any(selected < 0) or np.any(selected >= epoch_count):
        raise ValueError("epoch indices must be 1-based and within the data")
    return selected


def _selected_weights(weights: Any, selected: np.ndarray, epoch_count: int) -> np.ndarray | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0 or (values.size == 1 and (values[0] == 0 or np.isnan(values[0]))):
        return None
    if values.size != epoch_count:
        raise ValueError("weights must contain one value per input epoch")
    chosen = values[selected]
    if not np.all(np.isfinite(chosen)) or np.isclose(np.sum(chosen), 0):
        raise ValueError("selected weights must be finite and sum to a non-zero value")
    return chosen


__all__ = ["blockave"]

"""Low-level EEGLAB-compatible baseline removal."""

from __future__ import annotations

from typing import Any

import numpy as np


def rmbase(data: Any, frames: int | None = 0, basevector: Any = 0, *, return_mean: bool = False):
    """Subtract per-channel baseline means from continuous or epoched data.

    Args:
        data: Data array shaped ``(channels, frames * epochs)`` or
            ``(channels, frames, epochs)``.
        frames: Samples per epoch. ``0`` or ``None`` uses the full second
            dimension, matching EEGLAB's default.
        basevector: EEGLAB-style 1-based baseline frame indices, or ``0``/
            empty for the whole epoch.
        return_mean: When true, return ``(dataout, datamean)``.

    Returns:
        Baseline-corrected data, and optionally the channel-by-epoch means.
    """
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError("rmbase(): input data is empty")
    if array.ndim not in {2, 3}:
        raise ValueError("rmbase(): data must be 2D or 3D")

    original_shape = array.shape
    matrix = array.transpose(0, 2, 1).reshape(array.shape[0], -1) if array.ndim == 3 else array
    chans, total_frames = matrix.shape
    frames = int(frames or 0)
    if frames == 0:
        frames = total_frames
    if frames < 1 or frames > total_frames:
        raise ValueError("rmbase(): frames must be between 1 and the total number of samples")

    epochs = total_frames // frames
    if epochs < 1:
        raise ValueError("rmbase(): frames exceeds available data")
    if epochs * frames != total_frames:
        raise ValueError("rmbase(): total sample count must be an integer multiple of frames")

    baseline = _baseline_indices(basevector, frames)

    # Use float64 for mean calculation and output if input is integer; else preserve float precision.
    output_dtype = np.float64 if not np.issubdtype(matrix.dtype, np.floating) else matrix.dtype
    output = np.empty((chans, total_frames), dtype=output_dtype)

    # Reshape to (chans, epochs, frames) for vectorized mean calculation and subtraction.
    # Note: 'reshaped' and 'output_reshaped' are views.
    reshaped = matrix.reshape(chans, epochs, frames)
    output_reshaped = output.reshape(chans, epochs, frames)

    if baseline is None:
        means = np.nanmean(reshaped, axis=2, dtype=np.float64)
    else:
        # baseline contains 0-based indices within each epoch
        means = np.nanmean(reshaped[:, :, baseline], axis=2, dtype=np.float64)

    # Subtract means across all epochs. We use in-place operations on the output view
    # to avoid promoting float32 recordings to float64 temporaries during broadcasting.
    # means is (chans, epochs), we add a new axis for broadcasting: (chans, epochs, 1).
    np.copyto(output_reshaped, reshaped)
    output_reshaped -= means[:, :, np.newaxis].astype(output_dtype, copy=False)

    if array.ndim == 3:
        output = output.reshape(original_shape[0], original_shape[2], original_shape[1]).transpose(0, 2, 1)
    else:
        output = output.reshape(original_shape)
    return (output, means) if return_mean else output


def _baseline_indices(basevector: Any, frames: int) -> np.ndarray | None:
    values = _as_flat_values(basevector)
    if not values:
        return None
    if len(values) == 1 and int(values[0]) == 0:
        return None
    if len(values) == 1:
        raise ValueError("rmbase(): basevector should be 0 or a vector of frame indices")
    indices = np.asarray(values, dtype=int)
    if np.any(indices < 1):
        raise ValueError("rmbase(): basevector should contain positive EEGLAB frame indices")
    indices = indices[(indices >= 1) & (indices <= frames)]
    if indices.size == 0:
        raise ValueError("rmbase(): basevector does not overlap the epoch")
    return indices - 1


def _as_flat_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return []
        return [float(token) for token in text.replace(",", " ").split()]
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (list, tuple)):
        return list(np.asarray(value).ravel())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    return list(np.asarray(value).ravel())

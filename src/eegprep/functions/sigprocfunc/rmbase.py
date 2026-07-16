"""Low-level EEGLAB-compatible baseline removal."""

from __future__ import annotations

from typing import Any

import numpy as np

_NANMEAN_CHUNK_BYTES = 4 * 1024 * 1024


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
    channels = array.shape[0]
    total_frames = array.shape[1] * array.shape[2] if array.ndim == 3 else array.shape[1]
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

    # Keep frames contiguous, as in the original epoch loop, while using this
    # copy as the output buffer. Integer input retains the legacy float64 output.
    output_dtype = np.float64 if not np.issubdtype(array.dtype, np.floating) else array.dtype
    epoch_order = array.transpose(0, 2, 1) if array.ndim == 3 else array.reshape(channels, epochs, frames)
    output_reshaped = np.array(epoch_order, dtype=output_dtype, order="C", copy=True)
    means = np.empty((channels, epochs), dtype=np.result_type(array.dtype, np.float64))

    # np.nanmean makes a data copy and a validity mask. Process several epochs
    # at a time so those temporaries stay bounded without returning to a Python
    # loop per epoch.
    baseline_frames = frames if baseline is None else baseline.size
    mean_bytes_per_epoch = channels * baseline_frames * output_reshaped.dtype.itemsize
    chunk_epochs = max(1, min(epochs, _NANMEAN_CHUNK_BYTES // max(1, mean_bytes_per_epoch)))
    for start in range(0, epochs, chunk_epochs):
        stop = min(epochs, start + chunk_epochs)
        output_chunk = output_reshaped[:, start:stop, :]
        baseline_chunk = output_chunk if baseline is None else output_chunk[:, :, baseline]
        chunk_means = np.nanmean(baseline_chunk, axis=2, dtype=np.float64)
        means[:, start:stop] = chunk_means

        # Compute with the float64 means but write directly into the intended
        # output dtype. This preserves legacy float32 rounding without a
        # recording-sized float64 subtraction result.
        np.subtract(output_chunk, chunk_means[:, :, np.newaxis], out=output_chunk, casting="unsafe")

    if array.ndim == 3:
        output = output_reshaped.transpose(0, 2, 1)
    else:
        output = output_reshaped.reshape(original_shape)
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

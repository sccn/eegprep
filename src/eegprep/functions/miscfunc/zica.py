"""Baseline z-scoring and peak ordering of ICA activations."""

from __future__ import annotations

from typing import Any

import numpy as np


def zica(
    activations: Any,
    frames: int | None = None,
    baseline_frames: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score ICA activations from repeated within-epoch baseline frames.

    Args:
        activations: Components by concatenated samples matrix.
        frames: Samples per epoch, or the full data length for continuous data.
        baseline_frames: Zero-based frame indices within every epoch. At least
            three baseline frames are required.

    Returns:
        Z-scored activations, baseline standard deviations, maximum absolute
        z-scores, original zero-based component indices, and zero-based peak
        sample indices, all ordered by descending peak magnitude.
    """
    activity = np.asarray(activations, dtype=float)
    if activity.ndim != 2 or min(activity.shape) == 0:
        raise ValueError("activations must be a non-empty components-by-samples matrix")
    if not np.all(np.isfinite(activity)):
        raise ValueError("activations must contain only finite values")

    samples = activity.shape[1]
    epoch_frames = samples if frames in (None, 0) else int(frames)
    if epoch_frames < 1 or epoch_frames != frames and frames not in (None, 0):
        raise ValueError("frames must be a positive integer")
    if samples % epoch_frames:
        raise ValueError("frames must divide the activation sample count exactly")
    baseline = _baseline_indices(baseline_frames, epoch_frames)

    epoch_count = samples // epoch_frames
    epoch_activity = activity.reshape(activity.shape[0], epoch_count, epoch_frames)
    baseline_activity = epoch_activity[:, :, baseline].reshape(activity.shape[0], -1)
    baseline_sd = np.std(baseline_activity, axis=1, ddof=1)
    if np.any(baseline_sd == 0):
        components = np.flatnonzero(baseline_sd == 0).tolist()
        raise ValueError(f"baseline standard deviation is zero for components {components}")

    standardized = activity / baseline_sd[:, np.newaxis]
    peak_frames = np.argmax(np.abs(standardized), axis=1)
    peaks = np.abs(standardized[np.arange(standardized.shape[0]), peak_frames])
    order = np.argsort(peaks, kind="stable")[::-1]
    return standardized[order], baseline_sd[order], peaks[order], order, peak_frames[order]


def _baseline_indices(value: Any, frames: int) -> np.ndarray:
    if value is None or (np.isscalar(value) and value == 0):
        indices = np.arange(frames)
    else:
        raw = np.asarray(value)
        if raw.ndim != 1 or raw.size < 3 or not np.issubdtype(raw.dtype, np.number):
            raise ValueError("baseline_frames must contain at least three zero-based indices")
        numeric = raw.astype(float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError("baseline_frames must contain integer indices")
        indices = numeric.astype(int)
    if indices.size < 3:
        raise ValueError("baseline_frames must contain at least three indices")
    if np.any(indices < 0) or np.any(indices >= frames):
        raise ValueError(f"baseline_frames must be within 0..{frames - 1}")
    if np.unique(indices).size != indices.size:
        raise ValueError("baseline_frames must not contain duplicate indices")
    return indices


__all__ = ["zica"]

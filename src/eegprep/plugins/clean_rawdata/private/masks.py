"""Boolean-mask helpers for clean_rawdata ports."""

import numpy as np


def mask_to_intervals(mask: np.ndarray, *, value: bool) -> np.ndarray:
    """Convert a boolean sample mask to EEGLAB-style sample intervals."""
    target = np.asarray(mask, dtype=bool) == value
    if not np.any(target):
        return np.empty((0, 2), dtype=int)
    padded = np.concatenate([[False], target, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    return np.stack([starts, ends], axis=1).astype(int)

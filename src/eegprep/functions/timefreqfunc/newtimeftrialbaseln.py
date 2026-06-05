"""Single-trial baseline correction for EEGLAB-style ``newtimef``."""

from __future__ import annotations

from typing import Any

import numpy as np


def newtimeftrialbaseln(
    power: Any,
    timesout: Any,
    *,
    baseline: Any = 0,
    basenorm: str = "off",
    trialbase: str = "off",
) -> np.ndarray:
    """Apply trial-level divisive or standardized baseline correction."""
    corrected = np.asarray(power, dtype=float).copy()
    mode = str(trialbase).lower()
    if mode == "off":
        return corrected
    if mode not in {"on", "full"}:
        raise ValueError("trialbase must be 'on', 'off', or 'full'")
    base_indices = baseline_indices(timesout, baseline)
    time_axis = corrected.ndim - 2
    if mode == "full":
        base_indices = np.arange(corrected.shape[time_axis], dtype=int)
    if base_indices.size == 0:
        raise ValueError("No time points found in baseline")
    base = np.take(corrected, base_indices, axis=time_axis)
    mean = np.nanmean(base, axis=time_axis, keepdims=True)
    if str(basenorm).lower() == "on":
        ddof = 1 if base_indices.size > 1 else 0
        std = np.nanstd(base, axis=time_axis, ddof=ddof, keepdims=True)
        return (corrected - mean) / np.maximum(std, np.finfo(float).tiny)
    return corrected / np.maximum(mean, np.finfo(float).tiny)


def baseline_indices(timesout: Any, baseline: Any) -> np.ndarray:
    """Return zero-based EEGLAB baseline sample indices."""
    times = np.asarray(timesout, dtype=float).ravel()
    if baseline is None:
        values = np.asarray([0.0])
    else:
        values = np.asarray(baseline, dtype=float).reshape(-1)
    if values.size == 0:
        values = np.asarray([0.0])
    if values.size and np.isnan(values[0]):
        return np.arange(times.size, dtype=int)
    if values.size == 2:
        mask = (times >= values[0]) & (times <= values[1])
    elif values.size == 1:
        mask = times < values[0]
        if not np.any(mask):
            mask = np.ones(times.size, dtype=bool)
    else:
        ranges = values.reshape(-1, 2)
        mask = np.zeros(times.size, dtype=bool)
        for start, stop in ranges:
            mask |= (times >= start) & (times <= stop)
    indices = np.nonzero(mask)[0]
    if indices.size == 0:
        raise ValueError(
            "There are no sample points found in the baseline; disable baseline or change the baseline limits."
        )
    return indices.astype(int)


__all__ = ["baseline_indices", "newtimeftrialbaseln"]

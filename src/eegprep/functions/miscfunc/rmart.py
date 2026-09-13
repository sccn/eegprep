"""Legacy lagged-regression ocular artifact removal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.sigprocfunc.floatread import floatread
from eegprep.functions.sigprocfunc.floatwrite import floatwrite


_DEFAULT_THRESHOLD = 80.0
_EPOCH_FRAMES = 80
_LAG_COUNT = 40


def rmart(
    datafile: str | Path,
    outfile: str | Path,
    nchans: int,
    chanlist: Any,
    eogchan: Any,
    threshold: float = _DEFAULT_THRESHOLD,
    *,
    format: str | None = None,
) -> np.ndarray:
    """Remove threshold-triggered EOG artifacts from a float32 data file.

    ``chanlist`` and ``eogchan`` use EEGLAB-facing one-based channel numbers.
    The input and output are channel-major float32 matrices stored in MATLAB
    column order. The corrected selected channels are returned as well as
    written to ``outfile``.

    This implements the intended 40-lag local regression described by EEGLAB's
    legacy ``rmart`` rather than its currently unreachable processing branch.
    ICA or ASR is generally preferable for new analyses.
    """
    channel_count = _positive_integer(nchans, "nchans")
    selected = _channel_indices(chanlist, channel_count, "chanlist")
    eog_indices = _channel_indices(eogchan, channel_count, "eogchan")
    trigger = _DEFAULT_THRESHOLD if threshold == 0 else float(threshold)
    if not np.isfinite(trigger) or trigger < 0:
        raise ValueError("rmart: threshold must be a finite non-negative value")

    data = floatread(datafile, [channel_count, np.inf], format)
    eog = np.asarray(data[eog_indices], dtype=float)
    corrected = np.asarray(data[selected], dtype=float).copy()
    for output_index, channel_index in enumerate(selected):
        signal = np.asarray(data[channel_index], dtype=float)
        signal = signal - np.mean(signal)
        for center in _artifact_centers(signal, eog, trigger):
            signal = _regress_window(signal, eog, center)
        corrected[output_index] = signal

    floatwrite(corrected, outfile, format)
    return corrected


def _artifact_centers(signal: np.ndarray, eog: np.ndarray, threshold: float) -> list[int]:
    margin = _EPOCH_FRAMES // 2 + _LAG_COUNT // 2
    if signal.size <= 2 * margin:
        return []
    triggered = np.abs(signal) >= threshold
    triggered |= np.any(np.abs(eog) >= threshold, axis=0)
    candidates = np.flatnonzero(triggered)
    centers: list[int] = []
    for sample in candidates:
        center = int(np.clip(sample, margin, signal.size - margin))
        if not centers or center - centers[-1] >= _EPOCH_FRAMES:
            centers.append(center)
    return centers


def _regress_window(signal: np.ndarray, eog: np.ndarray, center: int) -> np.ndarray:
    half_epoch = _EPOCH_FRAMES // 2
    half_lags = _LAG_COUNT // 2
    signal_start = center - half_epoch
    signal_stop = center + half_epoch
    eog_start = signal_start - half_lags
    eog_stop = signal_stop + half_lags
    extended = eog[:, eog_start:eog_stop]
    columns = [np.ones(_EPOCH_FRAMES)]
    columns.extend(row[lag : lag + _EPOCH_FRAMES] for row in extended for lag in range(_LAG_COUNT))
    design = np.column_stack(columns)
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design,
        signal[signal_start:signal_stop],
        rcond=None,
    )
    output = signal.copy()
    output[signal_start:signal_stop] -= finite_matmul(design, coefficients)
    return output


def _channel_indices(values: Any, count: int, name: str) -> list[int]:
    numbers = np.asarray(values).reshape(-1)
    if numbers.size == 0:
        raise ValueError(f"rmart: {name} must not be empty")
    indices = [int(value) - 1 for value in numbers]
    if any(index + 1 != value or index < 0 or index >= count for index, value in zip(indices, numbers)):
        raise ValueError(f"rmart: {name} must contain one-based channel numbers")
    return indices


def _positive_integer(value: Any, name: str) -> int:
    result = int(value)
    if result != value or result < 1:
        raise ValueError(f"rmart: {name} must be a positive integer")
    return result


__all__ = ["rmart"]

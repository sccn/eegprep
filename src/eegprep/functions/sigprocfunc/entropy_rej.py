"""EEGLAB-compatible entropy rejection helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._rejection import parse_numeric_sequence
from eegprep.functions.sigprocfunc.realproba import realproba


def entropy_rej(
    signal: Any,
    threshold: Any = 0,
    oldentropy_rej: Any | None = None,
    normalize: int = 0,
    discret: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return entropy scores and rejection flags along the trial dimension."""
    if oldentropy_rej is not None and np.asarray(oldentropy_rej).size:
        entropy = np.asarray(oldentropy_rej, dtype=float)
    else:
        arr = _as_rows_points_trials(signal)
        entropy = np.zeros((arr.shape[0], arr.shape[2]), dtype=float)
        for row in range(arr.shape[0]):
            probabilities, _distribution = realproba(arr[row].reshape(-1, order="F"), discret)
            probabilities = np.asarray(probabilities, dtype=float).ravel()
            for trial in range(arr.shape[2]):
                trial_probabilities = probabilities[trial * arr.shape[1] : (trial + 1) * arr.shape[1]]
                entropy[row, trial] = -np.sum(
                    trial_probabilities * np.log(np.maximum(trial_probabilities, np.finfo(float).tiny))
                )
        if int(normalize):
            entropy = _normalize(entropy, signal_ndim=np.asarray(signal).ndim)
    return entropy, _threshold(entropy, threshold)


def _as_rows_points_trials(signal: Any) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.size, 1)
    elif arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    elif arr.ndim != 3:
        raise ValueError("signal must be 1-D, 2-D, or 3-D")
    return arr


def _normalize(scores: np.ndarray, *, signal_ndim: int) -> np.ndarray:
    if signal_ndim == 2:
        std = float(np.std(scores))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        return (scores - np.mean(scores)) / std
    std = np.std(scores, axis=1, keepdims=True)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (scores - scores.mean(axis=1, keepdims=True)) / std


def _threshold(scores: np.ndarray, threshold: Any) -> np.ndarray:
    values = parse_numeric_sequence(threshold, dtype=float)
    if not values or values[0] == 0:
        return np.zeros(scores.shape, dtype=bool)
    return np.abs(scores) > values[0]


__all__ = ["entropy_rej"]

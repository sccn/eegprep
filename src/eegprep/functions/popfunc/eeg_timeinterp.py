"""Interpolate a contiguous bad-sample interval."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline


def eeg_timeinterp(
    EEG: dict[str, Any],
    samples: Any,
    *,
    epochinds: Any | None = None,
    interpwin: int = 5,
    elecinds: Any | None = None,
    epochcont: str = "off",
) -> dict[str, Any]:
    """Spline-interpolate bad samples using neighboring activity.

    Samples and epochs use 1-based EEGLAB identifiers. Electrodes are 0-based
    so the result of :func:`eeg_chaninds` can be passed directly. ``samples``
    may be a ``[begin, end]`` range or a contiguous vector of bad samples.
    """
    output = copy.deepcopy(EEG)
    data = np.asarray(output["data"])
    was_continuous = data.ndim == 2
    if was_continuous:
        data = data[:, :, np.newaxis]
    if data.ndim != 3:
        raise ValueError("EEG.data must be two- or three-dimensional")
    sample_values = np.asarray(samples, dtype=int).ravel()
    if sample_values.size < 2:
        raise ValueError("samples must contain a begin and end sample")
    begin = int(sample_values.min())
    end = int(sample_values.max())
    pnts = data.shape[1]
    if begin < 1 or end > pnts:
        raise ValueError("samples must be 1-based and within EEG.pnts")
    if interpwin < 1:
        raise ValueError("interpwin must be positive")
    if epochcont not in {"on", "off"}:
        raise ValueError("epochcont must be 'on' or 'off'")
    electrodes = _zero_based_indices(elecinds, data.shape[0], "elecinds")
    epochs = _one_based_indices(epochinds, data.shape[2], "epochinds")
    width = end - begin
    margin = max(width * int(interpwin), 1)
    output_samples = np.arange(begin - 1, end)

    for epoch in epochs:
        for electrode in electrodes:
            signal = data[electrode, :, epoch]
            support_indices = np.concatenate(
                (
                    np.arange(max(begin - 1 - margin, 0), begin - 1),
                    np.arange(end, min(end + margin, pnts)),
                )
            )
            support_values = signal[support_indices]
            if epochcont == "on" and end + margin > pnts and epoch + 1 < data.shape[2]:
                extra_count = min(end + margin - pnts, pnts)
                support_indices = np.concatenate((support_indices, np.arange(pnts, pnts + extra_count)))
                support_values = np.concatenate((support_values, data[electrode, :extra_count, epoch + 1]))
            if support_indices.size < 2:
                raise ValueError("not enough neighboring samples for interpolation")
            data[electrode, output_samples, epoch] = CubicSpline(support_indices, support_values)(output_samples)
    output["data"] = data[:, :, 0] if was_continuous else data
    return output


def _one_based_indices(value: Any | None, length: int, name: str) -> list[int]:
    if value is None or np.asarray(value).size == 0:
        return list(range(length))
    values = np.asarray(value, dtype=int).ravel()
    if np.any(values < 1) or np.any(values > length):
        raise ValueError(f"{name} must be 1-based and within range")
    return (values - 1).tolist()


def _zero_based_indices(value: Any | None, length: int, name: str) -> list[int]:
    if value is None or np.asarray(value).size == 0:
        return list(range(length))
    values = np.asarray(value, dtype=int).ravel()
    if np.any(values < 0) or np.any(values >= length):
        raise ValueError(f"{name} must be 0-based and within range")
    return values.tolist()


__all__ = ["eeg_timeinterp"]

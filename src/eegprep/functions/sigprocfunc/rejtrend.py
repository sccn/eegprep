"""EEGLAB-compatible linear trend rejection helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def rejtrend(
    signal: Any,
    pointrange: int,
    maxslope: float,
    minstd: float,
    step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect trials containing linear trends with high slope and R-squared."""
    data = np.asarray(signal, dtype=float)
    if data.ndim != 3:
        raise ValueError("signal must be channels x points x trials")
    window = int(pointrange)
    if window < 2 or window > data.shape[1]:
        raise ValueError("pointrange must be between 2 and the number of points")
    stride = window if step is None else int(step)
    if stride < 1:
        raise ValueError("step must be positive")
    tolerance = 1000 * window * 1.1921e-7
    x = np.linspace(1 / window, 1, window)
    x_centered = x - np.mean(x)
    denom = float(np.sum(x_centered**2))
    reject_rows = np.zeros((data.shape[0], data.shape[2]), dtype=bool)
    for chan in range(data.shape[0]):
        for trial in range(data.shape[2]):
            for start in range(0, data.shape[1] - window + 1, stride):
                y = data[chan, start : start + window, trial]
                y_mean = float(np.mean(y))
                slope = float(np.sum((y - y_mean) * x_centered) / denom)
                if abs(slope) < float(maxslope):
                    continue
                intercept = y_mean - slope * float(np.mean(x))
                fitted = slope * x + intercept
                sst = max(float(np.sum((y - y_mean) ** 2)), tolerance)
                sse = float(np.sum((y - fitted) ** 2))
                if 1.0 - sse / sst > float(minstd):
                    reject_rows[chan, trial] = True
                    break
    return reject_rows.any(axis=0), reject_rows


__all__ = ["rejtrend"]

"""EEGLAB-compatible effective probability helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def realproba(data: Any, bins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample empirical probabilities and the distribution."""
    values = np.asarray(data, dtype=float)
    flat = values.ravel()
    if bins is None:
        bins = max(1, round(flat.size / 5))
    if flat.size == 0:
        return np.asarray(values, dtype=float), np.asarray([], dtype=float)
    if int(bins) <= 0:
        centered = flat - np.mean(flat)
        std = float(np.std(centered))
        if std == 0 or not np.isfinite(std):
            probabilities = np.ones(flat.shape, dtype=float) / flat.size
        else:
            z = centered / std
            probabilities = np.exp(-0.5 * z * z) / (2 * np.pi)
            probabilities = probabilities / np.sum(probabilities)
        return probabilities.reshape(values.shape), probabilities
    count = int(bins)
    minimum = float(np.nanmin(flat))
    maximum = float(np.nanmax(flat))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum == minimum:
        probabilities = np.ones(flat.shape, dtype=float)
        distribution = np.ones(count, dtype=float)
        distribution = distribution / distribution.sum()
        return probabilities.reshape(values.shape), distribution
    indices = np.floor((flat - minimum) / (maximum - minimum) * (count - 1)).astype(int)
    indices = np.clip(indices, 0, count - 1)
    distribution = np.bincount(indices, minlength=count).astype(float)
    probabilities = distribution[indices] / flat.size
    distribution = distribution / flat.size
    return probabilities.reshape(values.shape), distribution


__all__ = ["realproba"]

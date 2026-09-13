"""Legacy MATLAB-compatible sample quantiles."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.nan_std import _first_nonsingleton_axis


def quantile(data: Any, probabilities: Any, axis: int | None = None) -> np.ndarray:
    """Return quantiles using MATLAB's midpoint empirical-probability rule."""
    values = np.asarray(data, dtype=float)
    requested = np.asarray(probabilities, dtype=float).reshape(-1)
    if np.any((requested < 0) | (requested > 1)):
        raise ValueError("probabilities must lie between zero and one")
    if values.ndim == 0:
        return np.full(requested.shape, float(values))
    selected_axis = _first_nonsingleton_axis(values) if axis is None else axis
    moved = np.moveaxis(values, selected_axis, 0)
    if moved.shape[0] == 0:
        raise ValueError("data must not be empty")
    columns = moved.reshape(moved.shape[0], -1)
    output = np.empty((requested.size, columns.shape[1]), dtype=float)
    for column in range(columns.shape[1]):
        samples = np.sort(columns[~np.isnan(columns[:, column]), column])
        if samples.size == 0:
            output[:, column] = np.nan
            continue
        sample_probabilities = (np.arange(samples.size) + 0.5) / samples.size
        output[:, column] = np.interp(
            requested,
            sample_probabilities,
            samples,
            left=samples[0],
            right=samples[-1],
        )
    return output.reshape((requested.size, *moved.shape[1:]))


__all__ = ["quantile"]

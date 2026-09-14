"""Superimposed histogram plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def hist2(data1: Any, data2: Any, bins: Any = None, *, ax: Axes | None = None) -> Axes:
    """Plot two semi-transparent histograms on one Matplotlib axes.

    Vector ``bins`` values are bin centers, matching MATLAB ``hist`` rather
    than NumPy's bin-edge convention. Values beyond the first and last center
    are accumulated in the end bins.
    """
    first = _finite_vector(data1, "data1")
    second = _finite_vector(data2, "data2")
    centers = _bin_centers(first, second, bins)
    boundaries = np.concatenate(([-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]))
    first_counts = np.histogram(first, boundaries)[0]
    second_counts = np.histogram(second, boundaries)[0]
    widths = _bar_widths(centers)

    if ax is None:
        _figure, ax = plt.subplots()
    ax.bar(centers, first_counts, width=widths, color="blue", alpha=0.5, edgecolor="none", align="center")
    ax.bar(centers, second_counts, width=widths, color="red", alpha=0.5, edgecolor="none", align="center")
    ax.set_ylabel("Number of values")
    if centers.size > 1:
        ax.set_xlim(float(centers[0]), float(centers[-1]))
    return ax


def _finite_vector(data: Any, name: str) -> np.ndarray:
    values = np.asarray(data, dtype=float).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"hist2: {name} must contain finite values")
    return values


def _bin_centers(first: np.ndarray, second: np.ndarray, bins: Any) -> np.ndarray:
    minimum = min(float(np.min(first)), float(np.min(second)))
    maximum = max(float(np.max(first)), float(np.max(second)))
    if bins is None:
        count = 100
        return _even_centers(minimum, maximum, count)

    values = np.asarray(bins, dtype=float).reshape(-1)
    if values.size == 1:
        count = int(values[0])
        if count != values[0] or count < 1:
            raise ValueError("hist2: a scalar bins value must be a positive integer")
        return _even_centers(minimum, maximum, count)
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("hist2: bin centers must be finite and strictly increasing")
    return values


def _even_centers(minimum: float, maximum: float, count: int) -> np.ndarray:
    if minimum == maximum:
        if count == 1:
            return np.asarray([minimum])
        span = max(abs(minimum), 1.0) * 0.5
        minimum -= span
        maximum += span
    return np.linspace(minimum, maximum, count)


def _bar_widths(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        return np.asarray([1.0])
    differences = np.diff(centers)
    return np.concatenate(([differences[0]], np.minimum(differences[:-1], differences[1:]), [differences[-1]]))


__all__ = ["hist2"]

"""Two-dimensional Laplacian-of-Gaussian kernels."""

from __future__ import annotations

import numpy as np

from .gauss2d import _positive_shape, _threshold_magnitude


def laplac2d(
    rows: int,
    columns: int,
    sigma: float | None = None,
    mean_row: float | None = None,
    mean_column: float | None = None,
    cut: float = 0.0,
) -> np.ndarray:
    """Return EEGLAB's sampled two-dimensional Laplacian kernel."""
    row_count, column_count = _positive_shape(rows, columns)
    width = row_count / 5 if sigma is None else float(sigma)
    if width <= 0:
        raise ValueError("sigma must be positive")
    center_r = (row_count + 1) / 2 if mean_row is None else float(mean_row)
    center_c = (column_count + 1) / 2 if mean_column is None else float(mean_column)
    x, y = np.meshgrid(
        np.arange(1, row_count + 1, dtype=float),
        np.arange(1, column_count + 1, dtype=float),
        indexing="ij",
    )
    radius_squared = (x - center_r) ** 2 + (y - center_c) ** 2
    variance = width**2
    kernel = -np.exp(-0.5 * radius_squared / variance) * (radius_squared - variance) / variance**2
    return _threshold_magnitude(kernel, cut)


__all__ = ["laplac2d"]

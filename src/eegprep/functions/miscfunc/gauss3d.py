"""Three-dimensional Gaussian kernels."""

from __future__ import annotations

import numpy as np

from .gauss2d import _threshold_magnitude
from ._validation import integer_scalar


def gauss3d(
    rows: int,
    columns: int,
    depth: int,
    sigma_rows: float | None = None,
    sigma_columns: float | None = None,
    sigma_depth: float | None = None,
    mean_row: float | None = None,
    mean_column: float | None = None,
    mean_depth: float | None = None,
    cut: float = 0.0,
) -> np.ndarray:
    """Return an EEGLAB-compatible three-dimensional Gaussian kernel."""
    shape = (
        integer_scalar(rows, "rows"),
        integer_scalar(columns, "columns"),
        integer_scalar(depth, "depth"),
    )
    if min(shape) < 1:
        raise ValueError("kernel dimensions must be positive")
    sigmas = np.asarray(
        [
            shape[0] / 5 if sigma_rows is None else sigma_rows,
            shape[1] / 5 if sigma_columns is None else sigma_columns,
            shape[2] / 5 if sigma_depth is None else sigma_depth,
        ],
        dtype=float,
    )
    if np.any(sigmas <= 0):
        raise ValueError("standard deviations must be positive")
    means = np.asarray(
        [
            (shape[0] + 1) / 2 if mean_row is None else mean_row,
            (shape[1] + 1) / 2 if mean_column is None else mean_column,
            (shape[2] + 1) / 2 if mean_depth is None else mean_depth,
        ],
        dtype=float,
    )
    grids = np.meshgrid(
        *(np.arange(1, length + 1, dtype=float) for length in shape),
        indexing="ij",
    )
    exponent = sum(((grid - mean) / sigma) ** 2 for grid, mean, sigma in zip(grids, means, sigmas))
    kernel = np.exp(-0.5 * exponent) / (np.sqrt(np.prod(sigmas)) * np.pi)
    return _threshold_magnitude(kernel, cut)


__all__ = ["gauss3d"]

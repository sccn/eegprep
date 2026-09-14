"""Two-dimensional Gaussian kernels."""

from __future__ import annotations

import numpy as np

from ._validation import integer_scalar


def gauss2d(
    rows: int,
    columns: int,
    sigma_rows: float | None = None,
    sigma_columns: float | None = None,
    mean_row: float | None = None,
    mean_column: float | None = None,
    cut: float = 0.0,
) -> np.ndarray:
    """Return an EEGLAB-compatible two-dimensional Gaussian kernel.

    Coordinates intentionally remain one-based so explicit EEGLAB peak
    coordinates produce the same samples.
    """
    row_count, column_count = _positive_shape(rows, columns)
    sigma_r = row_count / 5 if sigma_rows is None else float(sigma_rows)
    sigma_c = column_count / 5 if sigma_columns is None else float(sigma_columns)
    if sigma_r <= 0 or sigma_c <= 0:
        raise ValueError("standard deviations must be positive")
    center_r = (row_count + 1) / 2 if mean_row is None else float(mean_row)
    center_c = (column_count + 1) / 2 if mean_column is None else float(mean_column)
    x, y = np.meshgrid(
        np.arange(1, row_count + 1, dtype=float),
        np.arange(1, column_count + 1, dtype=float),
        indexing="ij",
    )
    kernel = np.exp(-0.5 * (((x - center_r) / sigma_r) ** 2 + ((y - center_c) / sigma_c) ** 2))
    kernel /= np.sqrt(sigma_r * sigma_c) * np.pi
    return _threshold_magnitude(kernel, cut)


def _positive_shape(rows: int, columns: int) -> tuple[int, int]:
    shape = integer_scalar(rows, "rows"), integer_scalar(columns, "columns")
    if shape[0] < 1 or shape[1] < 1:
        raise ValueError("kernel dimensions must be positive")
    return shape


def _threshold_magnitude(values: np.ndarray, cut: float) -> np.ndarray:
    if not 0 <= cut <= 1:
        raise ValueError("cut must lie between zero and one")
    if cut == 0 or values.size == 0:
        return values
    output = values.copy()
    output[np.abs(output) < np.max(np.abs(output)) * cut] = 0
    return output


__all__ = ["gauss2d"]

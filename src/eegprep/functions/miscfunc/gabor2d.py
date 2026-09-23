"""Two-dimensional Gabor kernels."""

from __future__ import annotations

import numpy as np

from .gauss2d import _positive_shape, _threshold_magnitude


def gabor2d(
    rows: int,
    columns: int,
    frequency: float | None = None,
    angle: float = 0.0,
    sigma_rows: float | None = None,
    sigma_columns: float | None = None,
    mean_row: float | None = None,
    mean_column: float | None = None,
    phase: float = 0.0,
    cut: float = 0.0,
) -> np.ndarray:
    """Return a sinusoidal carrier under an anisotropic Gaussian envelope.

    ``frequency``, ``angle``, and ``phase`` are in degrees, preserving the
    EEGLAB contract. Magnitude thresholding retains both positive and negative
    lobes; the MATLAB implementation discards every negative value when
    ``cut`` is nonzero.
    """
    row_count, column_count = _positive_shape(rows, columns)
    sigma_r = row_count / 5 if sigma_rows is None else float(sigma_rows)
    sigma_c = column_count / 5 if sigma_columns is None else float(sigma_columns)
    if sigma_r <= 0 or sigma_c <= 0:
        raise ValueError("standard deviations must be positive")
    center_r = (row_count + 1) / 2 if mean_row is None else float(mean_row)
    center_c = (column_count + 1) / 2 if mean_column is None else float(mean_column)
    freq = 360 / row_count if frequency is None else float(frequency)
    x, y = np.meshgrid(
        np.arange(1, row_count + 1, dtype=float),
        np.arange(1, column_count + 1, dtype=float),
        indexing="ij",
    )
    rotated = ((x - center_r) + 1j * (y - center_c)) * np.exp(1j * np.deg2rad(angle))
    envelope = np.exp(-0.5 * (((x - center_r) / sigma_r) ** 2 + ((y - center_c) / sigma_c) ** 2))
    envelope /= np.sqrt(sigma_r * sigma_c) * np.pi
    kernel = np.sin(np.real(rotated) * np.deg2rad(freq) + np.deg2rad(phase)) * envelope
    return _threshold_magnitude(kernel, cut)


__all__ = ["gabor2d"]

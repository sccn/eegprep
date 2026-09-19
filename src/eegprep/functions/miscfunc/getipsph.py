"""In-place sphering and dimension-reducing quasi-sphering."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv


def getipsph(x: Any, m: int | None = None) -> np.ndarray:
    """Return an EEGLAB-compatible sphering or quasi-sphering matrix.

    Args:
        x: Channel-major data with shape ``(channels, samples)``.
        m: Output dimensionality. The default preserves all channels.

    Returns:
        A matrix with shape ``(m, channels)`` whose product with centered
        ``x`` has identity covariance when the retained covariance is full rank.
    """
    data = np.asarray(x, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError("getipsph: x must be a non-empty channels-by-samples matrix")
    if not np.isfinite(data).all():
        raise ValueError("getipsph: x must contain only finite values")

    channels, samples = data.shape
    dimensions = channels if m is None else _dimension(m, channels)
    centered = data - np.mean(data, axis=1, keepdims=True)
    covariance = finite_matmul(centered, centered.T) / samples
    covariance = (covariance + covariance.T) / 2
    left, singular_values, _right = np.linalg.svd(covariance)

    if dimensions == channels:
        root_inverse = finite_pinv(np.diag(np.sqrt(singular_values)))
        return finite_matmul(finite_matmul(left, root_inverse), left.T)

    variance_order = np.argsort(np.diag(covariance))[::-1]
    orientation_left, _values, orientation_right = np.linalg.svd(left[variance_order[:dimensions], :dimensions])
    root_inverse = finite_pinv(np.diag(np.sqrt(singular_values[:dimensions])))
    oriented = finite_matmul(orientation_left, orientation_right)
    whitened = finite_matmul(oriented, root_inverse)
    return finite_matmul(whitened, left[:, :dimensions].T)


def _dimension(value: Any, channels: int) -> int:
    dimensions = int(value)
    if dimensions != value or dimensions < 1 or dimensions > channels:
        raise ValueError("getipsph: m must be an integer from 1 through the number of channels")
    return dimensions


__all__ = ["getipsph"]

"""Principal-component compression."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import integer_scalar
from .misc import canonicalize_signs


def pcsquash(data: Any, components: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compress channel-major data into its leading principal components."""
    values = np.asarray(data)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("data must be numeric")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty channel-by-frame matrix")
    count = values.shape[0] if components is None else integer_scalar(components, "components")
    if count == 0:
        count = values.shape[0]
    if count < 1 or count > values.shape[0]:
        raise ValueError("components must lie between one and the channel count")
    data_mean = np.mean(values, axis=1)
    centered = values - data_mean[:, None]
    covariance = centered @ centered.conj().T / values.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = canonicalize_signs(eigenvectors[:, order])
    compressed = eigenvectors[:, :count].conj().T @ centered
    return eigenvectors, eigenvalues, compressed, data_mean


__all__ = ["pcsquash"]

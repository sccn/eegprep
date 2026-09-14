"""Oblique Promax rotation following an orthogonal Varimax rotation."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.miscfunc.runpca2 import _canonical_eigenvector_signs
from eegprep.functions.miscfunc.varimax import varimax


def promax(
    data: Any,
    n_components: int | None = None,
    max_iterations: int = 5,
    *,
    power: float = 4.0,
    tolerance: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return oblique Promax and orthogonal Varimax row rotations.

    ``n_components`` optionally projects centered input into its leading channel
    covariance subspace before rotation. Both returned matrices operate directly
    on the original rows.
    """
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("data must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("data must contain only finite values")
    rows = matrix.shape[0]
    count = rows if n_components in (None, 0) else int(n_components)
    if count < 1 or count > rows or (n_components not in (None, 0) and count != n_components):
        raise ValueError(f"n_components must be an integer from 1 to {rows}")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(power) or power <= 1:
        raise ValueError("power must be greater than 1")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")

    working, projection = _principal_projection(matrix, count)
    varimax_reduced, rotated = varimax(working, tolerance=tolerance, reorder=True)
    varimax_rotation = finite_matmul(varimax_reduced, projection)
    if count == 1:
        return varimax_rotation.copy(), varimax_rotation

    loadings = rotated.T
    rotated_axes = varimax_reduced.T
    previous_alignment: float | None = None
    for _iteration in range(int(max_iterations)):
        target = np.sign(loadings) * np.abs(loadings) ** power
        transform = np.linalg.lstsq(loadings, target, rcond=None)[0]
        norms = np.linalg.norm(transform, axis=0)
        transform = np.divide(transform, norms, out=np.zeros_like(transform), where=norms > 0)
        next_axes = finite_matmul(rotated_axes, transform)
        loadings = finite_matmul(loadings, transform)
        alignment = float(np.vdot(next_axes, rotated_axes).real)
        if previous_alignment is not None and abs(alignment - previous_alignment) < tolerance:
            rotated_axes = next_axes
            break
        previous_alignment = alignment
        rotated_axes = next_axes

    promax_reduced = rotated_axes.T
    return finite_matmul(promax_reduced, projection), varimax_rotation


def _principal_projection(data: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count == data.shape[0]:
        return data.copy(), np.eye(count)
    centered = data - np.mean(data, axis=1, keepdims=True)
    covariance = finite_matmul(centered, centered.T) / centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    vectors = _canonical_eigenvector_signs(eigenvectors[:, order[:count]])
    projection = vectors.T
    return finite_matmul(projection, centered), projection


__all__ = ["promax"]

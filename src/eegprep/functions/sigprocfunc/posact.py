"""Orient ICA components toward RMS-positive activations."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul


def posact(
    data: Any,
    weights: Any,
    sphere: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flip ICA components whose negative samples have the larger RMS.

    Args:
        data: Channels by samples input data.
        weights: Components by sphered-channels ICA weights.
        sphere: Sphering matrix, or ``None`` for identity.

    Returns:
        Reoriented activations, inverse unmixing matrix, and weights. All three
        outputs describe the same sign convention.
    """
    matrix = _finite_matrix(data, "data")
    weight_matrix = _finite_matrix(weights, "weights")
    sphere_matrix = np.eye(matrix.shape[0]) if sphere is None else _finite_matrix(sphere, "sphere")
    if sphere_matrix.shape[1] != matrix.shape[0]:
        raise ValueError("sphere columns must equal the number of data channels")
    if weight_matrix.shape[1] != sphere_matrix.shape[0]:
        raise ValueError("weights columns must equal the number of sphere rows")

    unmixing = finite_matmul(weight_matrix, sphere_matrix)
    activations = finite_matmul(unmixing, matrix)
    inverse = np.linalg.pinv(unmixing)
    orientation = np.ones(activations.shape[0])
    for component, activation in enumerate(activations):
        positive = activation[activation >= 0]
        negative = activation[activation < 0]
        positive_rms = _rms(positive)
        negative_rms = _rms(negative)
        if negative_rms > positive_rms:
            orientation[component] = -1

    oriented_activations = orientation[:, np.newaxis] * activations
    oriented_inverse = inverse * orientation[np.newaxis, :]
    oriented_weights = orientation[:, np.newaxis] * weight_matrix
    return oriented_activations, oriented_inverse, oriented_weights


def _finite_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _rms(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


__all__ = ["posact"]

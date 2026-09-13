"""Order ICA components by mean scalp-projected power."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul


def varsort(
    activations: Any,
    weights: Any,
    sphere: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return zero-based component order and descending projected power.

    The pseudoinverse supports PCA-reduced ICA decompositions, unlike the
    historical square-only implementation.
    """
    activity = _matrix(activations, "activations")
    weight_matrix = _matrix(weights, "weights")
    sphere_matrix = _matrix(sphere, "sphere")
    if activity.shape[0] != weight_matrix.shape[0]:
        raise ValueError("activations rows must equal the number of ICA components")
    if weight_matrix.shape[1] != sphere_matrix.shape[0]:
        raise ValueError("weights columns must equal sphere rows")
    if sphere_matrix.shape[1] < 2:
        raise ValueError("projected component variance requires at least two channels")

    inverse = np.linalg.pinv(finite_matmul(weight_matrix, sphere_matrix))
    projected_power = np.empty(activity.shape[0])
    for component in range(activity.shape[0]):
        projection = inverse[:, component, np.newaxis] * activity[component]
        sample_power = np.sum(projection * projection, axis=0) / (projection.shape[0] - 1)
        projected_power[component] = np.mean(sample_power)
    order = np.argsort(projected_power, kind="stable")[::-1]
    return order, projected_power[order]


def _matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


__all__ = ["varsort"]

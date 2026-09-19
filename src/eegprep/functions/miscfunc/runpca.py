"""Principal-component decomposition using singular-value decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np


def runpca(
    data: Any,
    n_components: int | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose channel-major data into principal components.

    Args:
        data: Variables by observations matrix.
        n_components: Number of principal dimensions, or all available dimensions.
        normalize: Put singular-value scaling in the components instead of the
            mixing matrix.

    Returns:
        Component time courses, mixing vectors, and a diagonal singular-value
        matrix. For centered input, ``mixing @ components`` reconstructs the
        retained-rank data.
    """
    matrix = _pca_matrix(data)
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    maximum = min(centered.shape)
    count = _component_count(n_components, maximum, matrix.shape[0])

    left, singular_values, right_transpose = np.linalg.svd(centered.T, full_matrices=False)
    left = left[:, :count]
    singular_values = singular_values[:count]
    right_transpose = right_transpose[:count]
    left, right_transpose = _canonical_svd_signs(left, right_transpose)

    if normalize:
        components = (left * singular_values).T
        mixing = right_transpose.T
    else:
        components = left.T
        mixing = right_transpose.T * singular_values
    return components, mixing, np.diag(singular_values)


def _pca_matrix(data: Any) -> np.ndarray:
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("data must be a non-empty variables-by-observations matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("data must contain only finite values")
    return matrix


def _component_count(value: int | None, maximum: int, variables: int) -> int:
    if value in (None, 0):
        return maximum
    if int(value) != value or not 1 <= int(value) <= maximum:
        if int(value) > variables:
            raise ValueError(f"n_components must not exceed the {variables} data rows")
        raise ValueError(f"n_components must be an integer from 1 to {maximum}")
    return int(value)


def _canonical_svd_signs(left: np.ndarray, right_transpose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    for component in range(right_transpose.shape[0]):
        anchor = int(np.argmax(np.abs(right_transpose[component])))
        if right_transpose[component, anchor] < 0:
            right_transpose[component] *= -1
            left[:, component] *= -1
    return left, right_transpose


__all__ = ["runpca"]

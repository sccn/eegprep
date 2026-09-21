"""Orthogonal Varimax rotation of component rows."""

from __future__ import annotations

from typing import Any

import numpy as np


def varimax(
    data: Any,
    tolerance: float = 1e-4,
    reorder: bool | str = True,
    *,
    max_iterations: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate matrix rows toward a sparse, orthogonal simple structure.

    Args:
        data: Components by features or observations matrix.
        tolerance: Pairwise rotation and convergence tolerance.
        reorder: Reorient and sort rows by descending energy. The strings
            ``"reorder"`` and ``"noreorder"`` are also accepted.
        max_iterations: Maximum sweeps through all row pairs.

    Returns:
        Orthogonal rotation and the rotated data ``rotation @ data``.
    """
    matrix = np.asarray(data, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("data must be a non-empty two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("data must contain only finite values")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    should_reorder = _reorder_flag(reorder)

    rotated = matrix.copy()
    rotation = np.eye(matrix.shape[0])
    previous = _criterion(rotated)
    no_improvement = 0

    for _iteration in range(int(max_iterations)):
        changed = False
        for first in range(rotated.shape[0] - 1):
            for second in range(first + 1, rotated.shape[0]):
                row_first = rotated[first]
                row_second = rotated[second]
                u_value = row_first * row_first - row_second * row_second
                v_value = 2 * row_first * row_second
                a_value = np.sum(u_value)
                b_value = np.sum(v_value)
                c_value = np.sum(u_value * u_value - v_value * v_value)
                d_value = np.sum(u_value * v_value)
                denominator = rotated.shape[1] * c_value + b_value**2 - a_value**2
                numerator = 2 * (rotated.shape[1] * d_value - a_value * b_value)
                if abs(numerator) <= tolerance * abs(denominator):
                    continue
                angle = 0.25 * np.arctan2(numerator, denominator)
                _rotate_rows(rotation, first, second, angle)
                _rotate_rows(rotated, first, second, angle)
                changed = True

        current = _criterion(rotated)
        if current == 0:
            relative_improvement = 0.0
        else:
            relative_improvement = (current - previous) / abs(current)
        no_improvement = no_improvement + 1 if relative_improvement <= tolerance else 0
        previous = current
        if not changed or no_improvement >= 2:
            break

    if should_reorder:
        signs = np.where(np.sum(rotated, axis=1) >= 0, 1.0, -1.0)
        rotation *= signs[:, np.newaxis]
        rotated *= signs[:, np.newaxis]
        order = np.argsort(np.sum(rotated * rotated, axis=1), kind="stable")[::-1]
        rotation = rotation[order]
        rotated = rotated[order]
    return rotation, rotated


def _criterion(data: np.ndarray) -> float:
    observations = data.shape[1]
    return float(np.sum(np.sum(data**4, axis=1) - np.sum(data**2, axis=1) ** 2 / observations))


def _rotate_rows(matrix: np.ndarray, first: int, second: int, angle: float) -> None:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    first_row = matrix[first].copy()
    second_row = matrix[second].copy()
    matrix[first] = cosine * first_row + sine * second_row
    matrix[second] = -sine * first_row + cosine * second_row


def _reorder_flag(value: bool | str) -> bool:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "reorder":
            return True
        if normalized == "noreorder":
            return False
        raise ValueError("reorder must be a boolean, 'reorder', or 'noreorder'")
    return bool(value)


__all__ = ["varimax"]

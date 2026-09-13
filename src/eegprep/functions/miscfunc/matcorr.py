"""Correlation-based row matching."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._validation import real_array


def matcorr(
    first: Any,
    second: Any,
    remove_mean: bool = False,
    method: int = 2,
    weighting: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match rows of two matrices by absolute cosine correlation.

    Methods are ``0`` (globally optimal Hungarian assignment), ``1`` (Vogel
    approximation), and ``2`` (successive maximum correlation). Returned row
    indices are zero-based. Rectangular inputs return ``min(rows)`` unique pairs.
    """
    left = real_array(first, "first")
    right = real_array(second, "second")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("input matrices must be 2-D with the same number of columns")
    if remove_mean:
        left = left - np.mean(left, axis=1, keepdims=True)
        right = right - np.mean(right, axis=1, keepdims=True)
    correlations = _cosine_rows(left, right)
    correlations = _apply_weighting(correlations, weighting)
    return _match_correlations(correlations, method)


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.outer(left_norm, right_norm)
    correlations = np.zeros((left.shape[0], right.shape[0]), dtype=float)
    np.divide(left @ right.T, denominator, out=correlations, where=denominator != 0)
    return correlations


def _apply_weighting(correlations: np.ndarray, weighting: Any | None) -> np.ndarray:
    if weighting is None:
        return correlations
    weights = real_array(weighting, "weighting")
    if weights.size == 0 or np.linalg.norm(weights) == 0:
        return correlations
    if weights.shape != correlations.shape:
        raise ValueError("weighting must have the same shape as the correlation matrix")
    return correlations * weights


def _match_correlations(correlations: np.ndarray, method: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if method == 0:
        rows, columns = linear_sum_assignment(-np.abs(correlations))
    elif method == 1:
        rows, columns = _vogel_pairs(np.abs(correlations))
    elif method == 2:
        rows, columns = _greedy_pairs(np.abs(correlations))
    else:
        raise ValueError("method must be 0, 1, or 2")
    values = correlations[rows, columns]
    order = np.argsort(-np.abs(values), kind="stable")
    return values[order], rows[order], columns[order], correlations


def _greedy_pairs(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active = scores.copy()
    row_indices: list[int] = []
    column_indices: list[int] = []
    for _ in range(min(active.shape)):
        flat_index = int(np.argmax(active))
        row, column = np.unravel_index(flat_index, active.shape)
        row_indices.append(int(row))
        column_indices.append(int(column))
        active[row, :] = -np.inf
        active[:, column] = -np.inf
    return np.asarray(row_indices), np.asarray(column_indices)


def _vogel_pairs(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    active_rows = list(range(scores.shape[0]))
    active_columns = list(range(scores.shape[1]))
    row_indices: list[int] = []
    column_indices: list[int] = []
    while active_rows and active_columns:
        submatrix = scores[np.ix_(active_rows, active_columns)]
        row_penalties = _penalties(submatrix, axis=1)
        column_penalties = _penalties(submatrix, axis=0)
        row_choice = int(np.argmax(row_penalties))
        column_choice = int(np.argmax(column_penalties))
        if row_penalties[row_choice] > column_penalties[column_choice]:
            chosen_row = row_choice
            chosen_column = int(np.argmax(submatrix[chosen_row]))
        else:
            chosen_column = column_choice
            chosen_row = int(np.argmax(submatrix[:, chosen_column]))
        row_indices.append(active_rows.pop(chosen_row))
        column_indices.append(active_columns.pop(chosen_column))
    return np.asarray(row_indices), np.asarray(column_indices)


def _penalties(scores: np.ndarray, axis: int) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=axis)
    if scores.shape[axis] == 1:
        return np.take(sorted_scores, -1, axis=axis).reshape(-1)
    return (np.take(sorted_scores, -1, axis=axis) - np.take(sorted_scores, -2, axis=axis)).reshape(-1)


__all__ = ["matcorr"]

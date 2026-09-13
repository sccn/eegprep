"""Minimum-cost bipartite assignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._validation import real_array


def hungarian(cost_matrix: Any) -> tuple[np.ndarray, float]:
    """Return a zero-based column-to-row assignment and its total cost."""
    costs = real_array(cost_matrix, "cost_matrix")
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost matrix must be square")
    if np.any(np.isnan(costs)) or np.any(np.isneginf(costs)):
        raise ValueError("cost matrix must not contain NaN or negative infinity")
    try:
        rows, columns = linear_sum_assignment(costs)
    except ValueError as error:
        raise ValueError("cost matrix has no finite complete assignment") from error
    assignment = np.empty(costs.shape[1], dtype=int)
    assignment[columns] = rows
    return assignment, float(costs[rows, columns].sum())


__all__ = ["hungarian"]

"""Minimum-cost bipartite assignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost_matrix: Any) -> tuple[np.ndarray, float]:
    """Return a zero-based column-to-row assignment and its total cost."""
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError("cost matrix must be square")
    if not np.isfinite(costs).all():
        raise ValueError("cost matrix must contain only finite values")
    rows, columns = linear_sum_assignment(costs)
    assignment = np.empty(costs.shape[1], dtype=int)
    assignment[columns] = rows
    return assignment, float(costs[rows, columns].sum())


__all__ = ["hungarian"]

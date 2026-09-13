"""Inverse permutations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import integer_array


def perminv(permutation: Any) -> np.ndarray:
    """Return the inverse of a zero-based permutation vector."""
    values = integer_array(permutation, "permutation")
    if values.ndim != 1:
        raise ValueError("permutation must be one-dimensional")
    if np.unique(values).size != values.size or np.any(np.sort(values) != np.arange(values.size)):
        raise ValueError("input must contain each integer from zero to n - 1 exactly once")
    inverse = np.empty_like(values)
    inverse[values] = np.arange(values.size)
    return inverse


__all__ = ["perminv"]

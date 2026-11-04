"""Dimension-wise random shuffling."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc._validation import integer_scalar


def shuffle(
    data: Any, dim: int | None = None, *, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle one 1-based dimension and return 1-based permutation indices.

    ``dim`` is EEGLAB's dimension, so 1 is the first axis. The default is the
    first dimension whose length is not 1, or the last dimension when every
    length is 1. EEGLAB only inspects the first three sizes and only shuffles
    dimensions 1 through 5. Subtract 1 before using the indices with NumPy:
    the forward indices rebuild the shuffled array and the inverse indices
    restore the original.
    """
    values = np.asarray(data)
    if values.ndim == 0:
        raise ValueError("data must have at least one dimension")
    if dim is None:
        selected_axis = _first_nonsingleton_axis(values)
    else:
        dimension = integer_scalar(dim, "dim")
        selected_axis = dimension - 1
        if selected_axis < 0 or selected_axis >= values.ndim:
            raise ValueError(f"dim {dimension} is out of range for {values.ndim}-D data")
    generator = np.random.default_rng() if rng is None else rng
    permutation = generator.permutation(values.shape[selected_axis])
    inverse = np.argsort(permutation)
    return np.take(values, permutation, axis=selected_axis), permutation + 1, inverse + 1


def _first_nonsingleton_axis(values: np.ndarray) -> int:
    for axis, length in enumerate(values.shape):
        if length != 1:
            return axis
    return values.ndim - 1


__all__ = ["shuffle"]

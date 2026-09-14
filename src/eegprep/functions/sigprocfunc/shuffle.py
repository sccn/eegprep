"""Axis-wise random shuffling."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc._validation import integer_scalar


def shuffle(
    data: Any, axis: int | None = None, *, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle one zero-based axis and return permutation and inverse indices."""
    values = np.asarray(data)
    if values.ndim == 0:
        raise ValueError("data must have at least one dimension")
    selected_axis = _first_nonsingleton_axis(values) if axis is None else integer_scalar(axis, "axis")
    if selected_axis < 0:
        selected_axis += values.ndim
    if selected_axis < 0 or selected_axis >= values.ndim:
        raise ValueError(f"axis {selected_axis} is out of range for {values.ndim}-D data")
    generator = np.random.default_rng() if rng is None else rng
    permutation = generator.permutation(values.shape[selected_axis])
    inverse = np.argsort(permutation)
    return np.take(values, permutation, axis=selected_axis), permutation, inverse


def _first_nonsingleton_axis(values: np.ndarray) -> int:
    for axis, length in enumerate(values.shape):
        if length != 1:
            return axis
    return values.ndim - 1


__all__ = ["shuffle"]

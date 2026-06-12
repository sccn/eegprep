"""Concatenate condition arrays along their case axis."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import condition_grid, flatten_grid


@dataclass(frozen=True)
class ConcatenatedData:
    """Data concatenated across condition case axes."""

    data: np.ndarray
    lengths: np.ndarray
    grid_shape: tuple[int, int]

    def __iter__(self) -> Iterator[np.ndarray | tuple[int, int]]:
        yield self.data
        yield self.lengths
        yield self.grid_shape


def concatdata(data: Any, *, axis: int = -1) -> ConcatenatedData:
    """Concatenate condition arrays along their case axis.

    Args:
        data: One- or two-dimensional sequence of condition arrays.
        axis: Axis in each condition array that stores cases.
    """

    grid = condition_grid(data, axis=axis, min_cases=1)
    arrays = flatten_grid(grid)
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")

    lengths = np.zeros(len(arrays) + 1, dtype=int)
    lengths[1:] = np.cumsum([array.shape[-1] for array in arrays])
    concatenated = np.concatenate(arrays, axis=-1)
    return ConcatenatedData(concatenated, lengths, (len(grid), len(grid[0])))


__all__ = ["ConcatenatedData", "concatdata"]

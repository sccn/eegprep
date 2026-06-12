"""Shared implementation helpers for EEGLAB-style statistics functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TwoWayEffects:
    """Row, column, and interaction values for a two-way design."""

    rows: Any
    columns: Any
    interaction: Any

    def __iter__(self):
        yield self.rows
        yield self.columns
        yield self.interaction


@dataclass(frozen=True)
class TwoWayAnovaResult:
    """Two-way ANOVA statistics and degrees of freedom."""

    rows: np.ndarray
    columns: np.ndarray
    interaction: np.ndarray
    df_rows: tuple[int, int]
    df_columns: tuple[int, int]
    df_interaction: tuple[int, int]

    def as_effects(self) -> TwoWayEffects:
        return TwoWayEffects(self.rows, self.columns, self.interaction)

    def df_effects(self) -> TwoWayEffects:
        return TwoWayEffects(self.df_rows, self.df_columns, self.df_interaction)


def as_numeric_array(value: Any, name: str, *, axis: int = -1, require_axis: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if require_axis and array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not require_axis:
        return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64, copy=False)

    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array with {array.ndim} dimensions")
    normalized_axis = axis % array.ndim
    if normalized_axis != array.ndim - 1:
        array = np.moveaxis(array, normalized_axis, -1)
    return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64, copy=False)


def condition_grid(data: Any, *, axis: int = -1, min_cases: int = 1) -> tuple[tuple[np.ndarray, ...], ...]:
    raw_grid = raw_condition_grid(data)
    grid: list[tuple[np.ndarray, ...]] = []
    for row_index, row in enumerate(raw_grid):
        converted_row = []
        for column_index, value in enumerate(row):
            array = as_numeric_array(value, f"condition ({row_index}, {column_index})", axis=axis)
            if array.shape[-1] < min_cases:
                raise ValueError(
                    f"condition ({row_index}, {column_index}) must have at least {min_cases} cases on the case axis"
                )
            converted_row.append(array)
        grid.append(tuple(converted_row))

    if len(grid) > 1 and len(grid[0]) == 1:
        grid = [tuple(row[0] for row in grid)]
    return tuple(grid)


def raw_condition_grid(data: Any) -> tuple[tuple[Any, ...], ...]:
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 1:
            return (tuple(data.tolist()),)
        if data.ndim == 2:
            return tuple(tuple(row) for row in data.tolist())
        raise ValueError("object-array condition grids must be one- or two-dimensional")
    if not isinstance(data, Sequence) or isinstance(data, str | bytes):
        raise TypeError("data must be a sequence of condition arrays")
    if len(data) == 0:
        raise ValueError("data must contain at least one condition")

    first = data[0]
    if looks_like_condition_row(first):
        rows = []
        expected_width: int | None = None
        for row in data:
            if not looks_like_condition_row(row):
                raise ValueError("data rows must all be condition sequences")
            row_tuple = tuple(row)
            if len(row_tuple) == 0:
                raise ValueError("condition rows must not be empty")
            if expected_width is None:
                expected_width = len(row_tuple)
            elif len(row_tuple) != expected_width:
                raise ValueError("all condition rows must have the same length")
            rows.append(row_tuple)
        return tuple(rows)
    return (tuple(data),)


def looks_like_condition_row(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, np.ndarray | str | bytes)


def flatten_grid(grid: tuple[tuple[np.ndarray, ...], ...]) -> list[np.ndarray]:
    return [array for row in grid for array in row]


def two_arrays(a: Any, b: Any | None, caller: str, *, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if b is None:
        grid = condition_grid(a, axis=axis, min_cases=1)
        arrays = flatten_grid(grid)
        if len(arrays) != 2:
            raise ValueError(f"{caller} requires exactly two arrays")
        return arrays[0], arrays[1]
    return (
        as_numeric_array(a, "a", axis=axis),
        as_numeric_array(b, "b", axis=axis),
    )


def one_way_arrays(data: Any, *, axis: int, paired: bool) -> list[np.ndarray]:
    grid = condition_grid(data, axis=axis, min_cases=2 if paired else 1)
    if len(grid) != 1:
        raise ValueError("one-way ANOVA helpers require a one-dimensional condition sequence")
    return list(grid[0])


def require_same_shapes(arrays: Sequence[np.ndarray], name: str) -> None:
    expected_shape = arrays[0].shape
    for index, array in enumerate(arrays):
        if array.shape != expected_shape:
            raise ValueError(f"{name} requires identical shapes; condition {index} has shape {array.shape}")


def two_way_stack(data: Any, *, axis: int, name: str) -> np.ndarray:
    grid = condition_grid(data, axis=axis, min_cases=2)
    if len(grid) < 1 or len(grid[0]) < 1:
        raise ValueError(f"{name} requires a non-empty condition grid")
    expected_shape = grid[0][0].shape
    for row_index, row in enumerate(grid):
        for column_index, array in enumerate(row):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} requires balanced cell shapes; condition ({row_index}, {column_index}) has "
                    f"shape {array.shape}, expected {expected_shape}"
                )
    return np.stack([np.stack(row, axis=-2) for row in grid], axis=-3)


def stat_mean(array: np.ndarray, *, axis: int) -> np.ndarray:
    result = np.mean(array, axis=axis)
    if np.iscomplexobj(array):
        return np.abs(result)
    return result


def stat_std(array: np.ndarray, *, axis: int) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.std(np.abs(array), axis=axis, ddof=1)
    return np.sqrt(np.sum((array - np.mean(array, axis=axis, keepdims=True)) ** 2, axis=axis) / (array.shape[axis] - 1))


def anova_values(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.abs(array)
    return array


def sum_square_residuals(array: np.ndarray, mean: np.ndarray, *, axis: int) -> np.ndarray:
    values = anova_values(array)
    return np.sum((values - np.expand_dims(mean, axis=axis)) ** 2, axis=axis)


def normalize_method(method: str) -> str:
    method_name = method.lower()
    if method_name == "parametric":
        return "param"
    if method_name == "permutation":
        return "perm"
    if method_name in {"param", "perm", "bootstrap"}:
        return method_name
    raise ValueError("method must be 'param', 'perm', 'permutation', 'parametric', or 'bootstrap'")


def paired_flag(grid: tuple[tuple[np.ndarray, ...], ...], paired: str | bool) -> bool:
    counts = [array.shape[-1] for array in flatten_grid(grid)]
    can_pair = len(set(counts)) == 1
    if paired == "auto":
        return can_pair
    if isinstance(paired, str):
        paired_name = paired.lower()
        if paired_name not in {"on", "off"}:
            raise ValueError("paired must be 'auto', 'on', 'off', True, or False")
        requested = paired_name == "on"
    else:
        requested = bool(paired)
    if requested and not can_pair:
        raise ValueError("paired statistics require the same number of cases in every condition")
    return requested


def effect_map(effect: Any, func: Any) -> Any:
    if isinstance(effect, TwoWayEffects):
        return TwoWayEffects(func(effect.rows), func(effect.columns), func(effect.interaction))
    return func(effect)


def rng_from_seed(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)

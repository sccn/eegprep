"""Match STUDY independent-variable values."""

from __future__ import annotations

from typing import Any

import numpy as np


def std_indvarmatch(value: Any, valuelist: Any) -> list[int]:
    """Return 1-based indices where an independent-variable value matches.

    This mirrors the useful EEGLAB behavior for strings, scalar numeric
    values, numeric vectors, and cell/list-valued variables while returning
    Python lists of EEGLAB-facing 1-based indices.
    """
    values = _as_list(valuelist)
    if all(isinstance(item, str) for item in values):
        queries = _as_list(value)
        return _unique_indices(
            index
            for query in queries
            for index, item in enumerate(values, start=1)
            if isinstance(query, str) and query == item
        )
    if values and all(_is_scalar_numeric(item) for item in values):
        queries = _as_list(value)
        return _unique_indices(
            index
            for query in queries
            for index, item in enumerate(values, start=1)
            if _is_scalar_numeric(query) and float(query) == float(item)
        )
    return [index for index, item in enumerate(values, start=1) if _equal_value(value, item)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_scalar_numeric(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 0 or value.size == 1
    return isinstance(value, (np.integer, np.floating, int, float)) and not isinstance(value, bool)


def _equal_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return list(_as_list(left)) == list(_as_list(right))
    return left == right


def _unique_indices(indices: Any) -> list[int]:
    output: list[int] = []
    for index in indices:
        index = int(index)
        if index not in output:
            output.append(index)
    return output


__all__ = ["std_indvarmatch"]

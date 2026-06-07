"""Select one subject from STUDY cached-measure arrays."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def std_selsubject(data: Any, subject: str, setinds: Any, allsubjects: Any, optndims: int | None = None) -> Any:
    """Return cached-measure cells with only the requested subject columns."""
    subjects = [str(item) for item in _as_list(allsubjects)]
    selected_subjects = [index for index, value in enumerate(subjects, start=1) if value.lower() == subject.lower()]
    if not selected_subjects:
        raise ValueError(f"Cannot select subject {subject} in list {subjects}")
    return _map_cells(data, setinds, set(selected_subjects), optndims)


def _map_cells(data: Any, setinds: Any, selected_subjects: set[int], optndims: int | None) -> Any:
    if isinstance(data, np.ndarray):
        return _select_array(data, setinds, selected_subjects, optndims)
    if isinstance(data, tuple):
        return tuple(
            _map_cells(item, _cell_at(setinds, index), selected_subjects, optndims) for index, item in enumerate(data)
        )
    if isinstance(data, list):
        return [
            _map_cells(item, _cell_at(setinds, index), selected_subjects, optndims) for index, item in enumerate(data)
        ]
    return deepcopy(data)


def _select_array(data: np.ndarray, setinds: Any, selected_subjects: set[int], optndims: int | None) -> np.ndarray:
    indices = [
        index for index, value in enumerate(np.asarray(setinds, dtype=int).ravel()) if int(value) in selected_subjects
    ]
    axis = max(0, int(optndims or data.ndim) - 1)
    if axis >= data.ndim:
        axis = data.ndim - 1
    return np.take(data, indices, axis=axis)


def _cell_at(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and index < len(value):
        return value[index]
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


__all__ = ["std_selsubject"]

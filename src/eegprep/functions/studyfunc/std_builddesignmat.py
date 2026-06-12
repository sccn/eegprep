"""Build EEGLAB-style STUDY design matrices."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import trialinfo_rows, value_matches


def std_builddesignmat(
    design: dict[str, Any], trialinfo: list[dict[str, Any]] | dict[str, Any], expanding: int | bool = False
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build a design matrix from a STUDY design and trial metadata.

    The output mirrors EEGLAB's helper: categorical factors are encoded as
    1-based level numbers, continuous factors keep their numeric values, and a
    constant column is appended at the end.
    """
    rows = trialinfo_rows(trialinfo)
    variables = [variable for variable in design.get("variable") or [] if isinstance(variable, dict)]
    variables = [variable for variable in variables if str(variable.get("label") or "") != "group"]
    matrix = np.full((len(rows), len(variables)), np.nan, dtype=float)
    labels = [str(variable.get("label") or "") for variable in variables]
    categorical = np.asarray(
        [str(variable.get("vartype") or "categorical").lower() == "categorical" for variable in variables],
        dtype=bool,
    )

    for column, variable in enumerate(variables):
        label = labels[column]
        if categorical[column]:
            levels = _levels(variable)
            for row_index, row in enumerate(rows):
                value = row.get(label)
                level_index = _matching_level(value, levels)
                if level_index is not None:
                    matrix[row_index, column] = float(level_index)
        else:
            for row_index, row in enumerate(rows):
                value = row.get(label)
                if value is not None and value != "":
                    matrix[row_index, column] = float(value)

    if bool(expanding):
        matrix, labels, categorical = _expand_categorical(matrix, labels, categorical, variables)

    constant = np.ones((matrix.shape[0], 1), dtype=float)
    matrix = np.hstack([matrix, constant])
    labels = [*labels, "constant"]
    categorical = np.asarray([*categorical.tolist(), False], dtype=bool)
    return matrix, labels, categorical


def _levels(variable: dict[str, Any]) -> list[Any]:
    values = variable.get("value")
    if isinstance(values, np.ndarray):
        values = values.tolist()
    if isinstance(values, (list, tuple)):
        return list(values)
    if values is None or values == "":
        return []
    return [values]


def _matching_level(value: Any, levels: list[Any]) -> int | None:
    for index, level in enumerate(levels, start=1):
        if _level_contains(level, value):
            return index
    return None


def _level_contains(level: Any, value: Any) -> bool:
    return value_matches(value, level)


def _expand_categorical(
    matrix: np.ndarray, labels: list[str], categorical: np.ndarray, variables: list[dict[str, Any]]
) -> tuple[np.ndarray, list[str], np.ndarray]:
    columns = []
    expanded_labels = []
    expanded_cat = []
    for column, label in enumerate(labels):
        if not categorical[column]:
            columns.append(matrix[:, column])
            expanded_labels.append(label)
            expanded_cat.append(False)
            continue
        levels = _levels(variables[column])
        for level_index, level in enumerate(levels, start=1):
            values = np.full(matrix.shape[0], np.nan, dtype=float)
            selected = matrix[:, column] == float(level_index)
            values[selected] = 1.0
            values[(~selected) & np.isfinite(matrix[:, column])] = 0.0
            columns.append(values)
            expanded_labels.append(f"{label}-{_level_label(level)}")
            expanded_cat.append(True)
    if not columns:
        return np.empty((matrix.shape[0], 0), dtype=float), [], np.asarray([], dtype=bool)
    return np.column_stack(columns), expanded_labels, np.asarray(expanded_cat, dtype=bool)


def _level_label(level: Any) -> str:
    if isinstance(level, np.ndarray):
        level = level.tolist()
    if isinstance(level, (list, tuple, set)):
        return "&".join(str(item) for item in level)
    return str(level)


__all__ = ["std_builddesignmat"]

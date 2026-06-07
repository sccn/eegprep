"""Return STUDY trial indices matching trialinfo queries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import parse_key_value_args


def std_gettrialsind(
    trialinfo: Any,
    *args: Any,
    return_values: bool = False,
    **kwargs: Any,
) -> Any:
    """Return 1-based trial indices matching all requested trialinfo values.

    ``trialinfo`` may be a list of trial dictionaries or a dictionary with a
    ``trialinfo`` field. MATLAB ``.mat`` filename loading is intentionally not
    part of the standalone runtime; pass loaded trialinfo rows instead.
    """
    if isinstance(trialinfo, (str, Path)):
        raise ValueError("std_gettrialsind expects loaded trialinfo rows, not a MATLAB filename")
    rows = _trial_rows(trialinfo)
    queries = parse_key_value_args(args, kwargs, lowercase_kwargs=False)
    if not queries:
        indices = list(range(1, len(rows) + 1))
        return (indices, []) if return_values else indices

    continuous_values: list[list[Any]] = []
    hits = np.ones(len(rows), dtype=bool)
    for label, expected in queries.items():
        field_values = [row.get(label) for row in rows]
        field_hits, values = _field_hits(field_values, expected, label)
        hits &= field_hits
        if values:
            continuous_values.append(values)
    indices = [index for index, keep in enumerate(hits, start=1) if keep]
    if not return_values:
        return indices
    selected_values = [[values[index - 1] for index in indices] for values in continuous_values]
    return indices, selected_values


def _field_hits(values: list[Any], expected: Any, label: str) -> tuple[np.ndarray, list[Any]]:
    if isinstance(expected, str) and (expected == "" or "<" in expected):
        numeric = [_numeric_or_none(value) for value in values]
        if expected == "":
            hits = np.asarray([value is not None for value in numeric], dtype=bool)
        else:
            bounds = _continuous_bounds(expected)
            hits = np.asarray(
                [value is not None and bounds[0] < value < bounds[1] for value in numeric],
                dtype=bool,
            )
        return hits, [np.nan if value is None else value for value in numeric]
    expected_values = _as_values(expected)
    hits = []
    for value in values:
        if isinstance(value, str):
            if any(not isinstance(item, str) for item in expected_values):
                raise ValueError(f"Type error: expected string values for field {label}")
        elif any(isinstance(item, str) for item in expected_values):
            raise ValueError(f"Type error: expected numerical values for field {label}")
        hits.append(any(_equal(value, item) for item in expected_values))
    return np.asarray(hits, dtype=bool), []


def _continuous_bounds(text: str) -> tuple[float, float]:
    if text.count("<") != 1 or "=" in text:
        raise ValueError("continuous ranges must use strict 'lower<upper' syntax")
    lower, upper = text.split("<", 1)
    return float(lower), float(upper)


def _numeric_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        raise ValueError("continuous trialinfo queries require numerical field values")
    return float(value)


def _trial_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and "trialinfo" in value:
        value = value.get("trialinfo")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _as_values(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    return left == right


__all__ = ["std_gettrialsind"]

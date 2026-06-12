"""Shared EEGLAB-style value parsing helpers."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

import numpy as np


_TOKEN_PATTERN = re.compile(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)")
_RANGE_TOKEN = re.compile(r"^([^:]+):([^:]+)(?::([^:]+))?$")


def parse_key_value_args(
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    lowercase_keys: bool = True,
    lowercase_kwargs: bool = False,
) -> dict[str, Any]:
    """Parse EEGLAB-style key/value positional arguments."""
    if len(args) % 2:
        raise ValueError("Key/value arguments must be in pairs")
    options: dict[str, Any] = {}
    for key, value in (kwargs or {}).items():
        parsed_key = str(key).lower() if lowercase_kwargs else key
        options[parsed_key] = value
    for index in range(0, len(args), 2):
        key = args[index]
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        parsed_key = key.lower() if lowercase_keys else key
        options[parsed_key] = args[index + 1]
    return options


def parse_text_tokens(text: Any, *, parse_ints: bool = False) -> list[Any]:
    """Parse MATLAB text/cell-list token strings used by GUI dialogs."""
    tokens = _TOKEN_PATTERN.findall(str(text).strip().strip("{}"))
    values = [next(part for part in token if part) for token in tokens]
    if not parse_ints:
        return values
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def parse_numeric_sequence(value: Any, *, dtype: type = float) -> list[Any]:
    """Parse EEGLAB-style numeric vectors, including ``start:stop`` ranges."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [dtype(item) for item in value.ravel().tolist()]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [dtype(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        values: list[Any] = []
        for item in value:
            if isinstance(item, str):
                values.extend(parse_numeric_sequence(item, dtype=dtype))
            else:
                values.append(dtype(item))
        return values
    text = str(value).strip().strip("[]")
    if not text:
        return []
    values = []
    for token in re.split(r"[\s,;]+", text):
        if not token:
            continue
        match = _RANGE_TOKEN.match(token)
        if match:
            values.extend(_parse_range_token(match, dtype=dtype))
        else:
            values.append(dtype(_parse_numeric_atom(token)))
    return values


def is_empty_value(value: Any) -> bool:
    """Return whether a GUI/history value means empty in EEGLAB dialogs."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().strip("[]{}") == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple, set, dict)) and len(value) == 0


def is_on(value: Any) -> bool:
    """Normalize EEGLAB-style on/off values."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "on", "true", "yes"}
    if isinstance(value, np.ndarray):
        return bool(value.size and np.asarray(value).ravel()[0])
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        values = list(value)
        return bool(values and values[0])
    return bool(value)


def _parse_range_token(match: re.Match[str], *, dtype: type) -> list[Any]:
    first = _parse_numeric_atom(match.group(1))
    second = _parse_numeric_atom(match.group(2))
    third = match.group(3)
    if third is None:
        start, stop = first, second
        step = 1.0 if stop >= start else -1.0
    else:
        start, step, stop = first, second, _parse_numeric_atom(third)
    if step == 0 or not np.all(np.isfinite([start, step, stop])):
        raise ValueError("Invalid colon range")
    if (stop - start) * step < 0:
        return []
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = [start + index * step for index in range(max(count, 0))]
    if values and np.isclose(values[-1], stop, rtol=0.0, atol=max(abs(step), 1.0) * 1e-12):
        values[-1] = stop
    return [dtype(value) for value in values]


def _parse_numeric_atom(value: Any) -> float:
    text = str(value).strip()
    if text.lower() == "nan":
        return float("nan")
    if text.lower() == "inf":
        return float("inf")
    if text.lower() == "-inf":
        return float("-inf")
    return float(text)

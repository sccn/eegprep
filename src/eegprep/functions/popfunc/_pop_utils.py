"""Shared helpers for EEGLAB-style pop function argument parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np


_TOKEN_PATTERN = re.compile(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)")


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
    """Parse MATLAB text/cell-list token strings used by pop dialogs."""
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


def format_history_value(
    value: Any,
    *,
    bool_style: str | None = None,
    cell_for_sequence: str | None = "all_strings",
    string_separator: str = " ",
    empty_sequence: str = "[]",
    none_as_empty: bool = False,
    dict_formatter: Callable[[list[dict[str, Any]]], str] | None = None,
    number_formatter: Callable[[Any], str] | None = None,
) -> str:
    """Format a Python value as an EEGLAB command-history literal."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Path):
        value = str(value)
    if value is None and none_as_empty:
        return "[]"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if bool_style == "onoff" and isinstance(value, bool):
        return "'on'" if value else "'off'"
    if isinstance(value, dict) and dict_formatter is not None:
        return dict_formatter([value])
    if isinstance(value, (list, tuple)):
        values = list(value)
        if not values:
            return empty_sequence
        if dict_formatter is not None and all(isinstance(item, dict) for item in values):
            return dict_formatter(values)
        if any(_is_nested_sequence(item) for item in values):
            return "[" + "; ".join(_format_history_row(item, number_formatter) for item in values) + "]"
        if _sequence_should_use_cell(values, cell_for_sequence):
            return "{" + string_separator.join(
                format_history_value(
                    item,
                    bool_style=bool_style,
                    cell_for_sequence=cell_for_sequence,
                    string_separator=string_separator,
                    empty_sequence=empty_sequence,
                    none_as_empty=none_as_empty,
                    dict_formatter=dict_formatter,
                    number_formatter=number_formatter,
                )
                for item in values
            ) + "}"
        return "[" + " ".join(_format_history_number(item, number_formatter) for item in values) + "]"
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, (int, float)):
        return _format_history_number(value, number_formatter)
    return str(value)


def _is_nested_sequence(value: Any) -> bool:
    return isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)))


def _format_history_row(value: Any, formatter: Callable[[Any], str] | None) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return " ".join(_format_history_number(item, formatter) for item in value)
    return _format_history_number(value, formatter)


def _sequence_should_use_cell(values: list[Any], mode: str | None) -> bool:
    if mode == "always":
        return True
    if mode == "all_strings":
        return all(isinstance(item, str) for item in values)
    if mode == "any_strings":
        return any(isinstance(item, str) for item in values)
    return False


def _format_history_number(value: Any, formatter: Callable[[Any], str] | None) -> str:
    if formatter is not None:
        return formatter(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, str):
        return format_history_value(value)
    if isinstance(value, float):
        if np.isneginf(value):
            return "-Inf"
        if np.isposinf(value):
            return "Inf"
        if value.is_integer():
            return str(int(value))
    return str(value)

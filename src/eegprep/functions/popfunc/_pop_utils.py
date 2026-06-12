"""Shared helpers for EEGLAB-style pop function argument parsing."""

from __future__ import annotations

from pathlib import PurePath
from typing import Any, Callable

import numpy as np

from eegprep.functions.miscfunc.value_parsing import (
    is_empty_value as is_empty_value,
    is_on as is_on,
    parse_key_value_args as parse_key_value_args,
    parse_numeric_sequence as parse_numeric_sequence,
    parse_text_tokens as parse_text_tokens,
)


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
    if isinstance(value, PurePath):
        value = value.as_posix()
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
            return (
                "{"
                + string_separator.join(
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
                )
                + "}"
            )
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

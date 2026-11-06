"""Format arguments for replayable EEGLAB command history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import _matlab_num2str
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence


def vararg2str(
    arguments: Any,
    inputnames: Sequence[str] | None = None,
    inputnum: Sequence[int] | None = None,
    nostrconv: Sequence[int | bool] | None = None,
) -> str:
    """Return a comma-separated EEGLAB expression for function arguments.

    ``inputnames`` can preserve caller variable names instead of serializing
    their values. ``nostrconv`` leaves selected string expressions unquoted.
    ``inputnum`` is accepted for compatibility with EEGLAB, whose current
    implementation likewise does not use it during conversion.
    """
    del inputnum
    values = list(arguments) if isinstance(arguments, (list, tuple)) else [arguments]
    names = _padded(inputnames, len(values), "")
    raw_strings = _padded(nostrconv, len(values), False)
    rendered = []
    for value, name, raw_string in zip(values, names, raw_strings):
        if name:
            rendered.append(str(name))
        else:
            rendered.append(_format_value(value, raw_string=bool(raw_string)))
    return ",".join(rendered)


def _matlab_history_value(value: Any) -> str:
    if isinstance(value, str) and not _looks_numeric(value):
        return f"'{value}'"
    values = parse_numeric_sequence(value, dtype=float)
    if not values:
        return "[]"
    if len(values) == 1:
        return _matlab_num2str(values[0])
    return "[" + _num2str_contarray(values) + "] "


def _looks_numeric(value: str) -> bool:
    token = value.strip().strip("[]")
    if not token:
        return True
    return all(character.isdigit() or character in ".:-+eE,; " for character in token)


def _num2str_contarray(values: list[float]) -> str:
    # vararg2str.m:196-242
    array = [float(item) for item in values]
    if len(array) == 2:
        return f"{_matlab_num2str(array[0])} {_matlab_num2str(array[1])}"
    if any(not item.is_integer() for item in array):
        return " ".join(_matlab_num2str(item) for item in array)
    rendered = _matlab_num2str(array[0])
    skip = 0
    indent = array[1] - array[0]
    for index in range(1, len(array)):
        if array[index] != array[index - 1] + indent or indent == 0:
            if skip <= 1:
                if skip == 0:
                    rendered = f"{rendered} {_matlab_num2str(array[index])}"
                else:
                    rendered = f"{rendered} {_matlab_num2str(array[index - 1])} {_matlab_num2str(array[index])}"
            elif indent == 1:
                rendered = f"{rendered}:{_matlab_num2str(array[index - 1])} {_matlab_num2str(array[index])}"
            else:
                rendered = (
                    f"{rendered}:{_matlab_num2str(indent)}:{_matlab_num2str(array[index - 1])} "
                    f"{_matlab_num2str(array[index])}"
                )
            skip = 0
            indent = array[index] - array[index - 1]
        else:
            skip += 1
    if array[-1] == array[-2] + indent and skip != 0:
        if indent == 1:
            rendered = f"{rendered}:{_matlab_num2str(array[-1])}"
        elif indent == 0:
            rendered = f"{rendered} {_matlab_num2str(array[-1])}"
        else:
            rendered = f"{rendered}:{_matlab_num2str(indent)}:{_matlab_num2str(array[-1])}"
    return rendered


def _padded(values: Sequence[Any] | None, length: int, default: Any) -> list[Any]:
    result = [] if values is None else list(values)
    return (result + [default] * length)[:length]


def _format_value(value: Any, *, raw_string: bool = False) -> str:
    if isinstance(value, str):
        return value if raw_string else format_history_value(value)
    if isinstance(value, dict):
        fields: list[str] = []
        for name, field_value in value.items():
            fields.extend((format_history_value(str(name)), _format_value(field_value)))
        return f"struct({','.join(fields)})" if fields else "struct([])"
    if isinstance(value, (list, tuple)):
        return "{" + vararg2str(value) + "}"
    if isinstance(value, np.ndarray):
        return format_history_value(value, cell_for_sequence=None)
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return format_history_value(value, cell_for_sequence=None)


def matlab_array2str(values: Any) -> str:
    """Format one numeric array the way EEGLAB ``vararg2str`` formats it."""
    array = np.asarray([] if values is None else values, dtype=float)
    if array.size == 0:
        return "[]"
    if array.ndim == 0 or array.size == 1:
        return _matlab_num(float(array.reshape(-1)[0]))
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] == 1:
        return f"[{_matlab_contarray(array.reshape(-1))}] "
    if array.shape[1] == 1:
        return f"[{_matlab_contarray(array.reshape(-1))}]' "
    return "[" + ";".join(_matlab_contarray(row) for row in array) + "]"


def _matlab_contarray(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=float).reshape(-1)
    if values.size == 1:
        return _matlab_num(float(values[0]))
    if values.size == 2:
        return f"{_matlab_num(float(values[0]))} {_matlab_num(float(values[1]))}"
    if np.all(np.isnan(values) | np.isclose(values, np.rint(values), equal_nan=True)):
        return _matlab_integer_row(values)
    return " ".join(_matlab_num(float(value)) for value in values)


def _matlab_integer_row(values: np.ndarray) -> str:
    numbers = [float(value) for value in values]
    parts = [_matlab_num(numbers[0])]
    skip = 0
    indent = numbers[1] - numbers[0]
    last_index = 1
    for index in range(1, len(numbers)):
        last_index = index
        if numbers[index] != numbers[index - 1] + indent or indent == 0:
            if skip <= 1:
                if skip == 0:
                    parts.append(_matlab_num(numbers[index]))
                else:
                    parts.append(_matlab_num(numbers[index - 1]))
                    parts.append(_matlab_num(numbers[index]))
            elif indent == 1:
                parts[-1] = f"{parts[-1]}:{_matlab_num(numbers[index - 1])}"
                parts.append(_matlab_num(numbers[index]))
            else:
                parts[-1] = f"{parts[-1]}:{_matlab_num(indent)}:{_matlab_num(numbers[index - 1])}"
                parts.append(_matlab_num(numbers[index]))
            skip = 0
            indent = numbers[index] - numbers[index - 1]
        else:
            skip += 1
    if numbers[last_index] == numbers[last_index - 1] + indent and skip != 0:
        if indent == 1:
            parts[-1] = f"{parts[-1]}:{_matlab_num(numbers[last_index])}"
        elif indent == 0:
            parts.append(_matlab_num(numbers[last_index]))
        else:
            parts[-1] = f"{parts[-1]}:{_matlab_num(indent)}:{_matlab_num(numbers[last_index])}"
    return " ".join(parts)


def _matlab_num(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "Inf"
    if np.isneginf(value):
        return "-Inf"
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), ".16g")

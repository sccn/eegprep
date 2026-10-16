"""Extract values from EEGLAB command-history strings."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np


def getkeyval(command: str, variable: str | int, mode: str | int | Sequence[int] = "", default: Any = "") -> Any:
    """Return an argument from an EEGLAB command-history expression.

    String ``variable`` values address key/value arguments. Whole numbers use
    EEGLAB-facing 1-based argument positions. ``mode="present"`` returns a
    boolean-like integer, ``mode="full"`` includes the key in the returned
    command fragment, and whole-number modes select 1-based elements from a
    numeric vector value.
    """
    if not command:
        return default
    arguments = _split_arguments(command)
    position = _whole_numbers(variable)
    if position is not None and len(position) == 1:
        index = position[0] - 1
        return arguments[index].strip() if 0 <= index < len(arguments) else default

    key_index = _find_key(arguments, str(variable))
    if isinstance(mode, str) and mode.lower() == "present":
        return int(key_index is not None)
    if key_index is None:
        if isinstance(mode, str) and mode.lower() == "full":
            parent_index = next((index for index, argument in enumerate(arguments) if variable in argument), None)
            if parent_index is not None and parent_index + 1 < len(arguments):
                return f"'{variable}', {arguments[parent_index + 1]}"
        return default
    if key_index + 1 >= len(arguments):
        return default

    raw_value = arguments[key_index + 1].strip()
    if isinstance(mode, str) and mode.lower() == "full":
        return f"'{variable}', {raw_value}"
    indices = _whole_numbers(mode)
    if indices is None and isinstance(mode, Sequence) and not isinstance(mode, (str, bytes)):
        indices = [int(index) for index in mode]
    if indices is not None:
        return _select_numeric_value(raw_value, indices, default)
    return _unquote(raw_value)


def _split_arguments(command: str) -> list[str]:
    open_paren = command.find("(")
    close_paren = command.rfind(")")
    if open_paren < 0 or close_paren < open_paren:
        return []
    body = command[open_paren + 1 : close_paren]
    arguments: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    quoted = False
    index = 0
    while index < len(body):
        char = body[index]
        if char == "'":
            if quoted and index + 1 < len(body) and body[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif not quoted and char in depths:
            depths[char] += 1
        elif not quoted and char in closing:
            depths[closing[char]] -= 1
        elif not quoted and char == "," and not any(depths.values()):
            arguments.append(body[start:index])
            start = index + 1
        index += 1
    if body[start:].strip():
        arguments.append(body[start:])
    return arguments


def _whole_numbers(value: Any) -> list[int] | None:
    if isinstance(value, (bool, str, bytes)):
        return None
    if isinstance(value, Integral):
        return [int(value)]
    if isinstance(value, Real):
        number = float(value)
        return [int(number)] if number.is_integer() else None
    if isinstance(value, np.ndarray) and value.dtype.kind in "iuf":
        flat = np.asarray(value, dtype=float).ravel()
        if flat.size == 0 or not np.all(np.isfinite(flat)) or not np.all(flat == np.trunc(flat)):
            return None
        return [int(item) for item in flat]
    return None


def _find_key(arguments: list[str], variable: str) -> int | None:
    for index in range(0, len(arguments) - 1):
        if _unquote(arguments[index].strip()) == variable:
            return index
    return None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def _select_numeric_value(raw_value: str, indices: list[int], default: Any) -> Any:
    if not indices:
        return raw_value
    text = raw_value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values = np.fromstring(text.replace(",", " "), sep=" ")
    available = [index for index in indices if 1 <= index <= values.size]
    if not available:
        return default
    selected = values[np.asarray(available, dtype=int) - 1]
    return " ".join(_format_number(value) for value in selected)


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))

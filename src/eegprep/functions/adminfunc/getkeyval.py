"""Extract values from EEGLAB command-history strings."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import matlab_ints, scalar_index
from eegprep.functions.miscfunc.matlab_outputs import as_row


def getkeyval(command: str, variable: Any, mode: Any = "", default: Any = "") -> Any:
    """Return an argument from an EEGLAB command-history expression.

    String ``variable`` values address key/value arguments. Integers,
    whole-number floats, and 1-by-1 numeric arrays use EEGLAB-facing 1-based
    argument positions, because MATLAB passes those positions as doubles.
    ``mode="present"`` returns a boolean-like integer, ``mode="full"`` includes
    the key in the returned command fragment, and integer modes select 1-based
    elements from a numeric vector value. A whole-number vector is the same
    kind of mode.
    """
    if not command:
        return default
    arguments = _split_arguments(command)
    position = scalar_index(variable)
    if position is not None:
        index = position - 1
        return arguments[index].strip() if 0 <= index < len(arguments) else default

    key_index = _find_key(arguments, str(variable))
    if isinstance(mode, str) and mode.lower() == "present":
        return as_row(float(key_index is not None))
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
    indices = _mode_indices(mode)
    if indices is not None:
        return _select_numeric_value(raw_value, indices, default)
    return _unquote(raw_value)


def _mode_indices(mode: Any) -> list[int] | None:
    if isinstance(mode, (str, bytes)):
        return None
    if isinstance(mode, np.ndarray) and (mode.dtype == object or mode.size == 0):
        return None
    indices = matlab_ints(mode)
    if not indices:
        return None
    return indices


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

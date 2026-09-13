"""Format arguments for replayable EEGLAB command history."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value


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

"""Shared pop_* helpers for bundled firfilt dialogs and history."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


def normalize_pop_options(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    positional: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Parse Python keyword, EEGLAB key/value, and legacy positional options."""
    if args and isinstance(args[0], str):
        return parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True)
    options = {str(key).lower(): value for key, value in kwargs.items()}
    for key, value in zip(positional, args):
        options[key] = value
    return options


def bool_value(value: Any, *, default: bool = False) -> bool:
    """Return an EEGLAB-compatible boolean option value."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        value = value.ravel()[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def numeric_or_none(value: Any) -> float | None:
    """Return a scalar float, treating EEGLAB empty/zero-ish values as ``None``."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return None
        value = text
    return float(value)


def int_or_none(value: Any) -> int | None:
    """Return a scalar int or ``None`` for EEGLAB empty values."""
    number = numeric_or_none(value)
    return None if number is None else int(number)


def vector_or_none(value: Any) -> list[float] | None:
    """Parse an EEGLAB numeric vector option."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return None
        return [float(token) for token in text.replace(",", " ").split()]
    if isinstance(value, np.ndarray):
        values = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(np.asarray(value).ravel())
    else:
        values = [value]
    if not values:
        return None
    return [float(item) for item in values]


def has_value(value: Any) -> bool:
    """Return whether an EEGLAB option carries a non-empty value."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip().strip("[]"))
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, set)):
        return len(value) > 0
    return True


def history_command(function_name: str, options: dict[str, Any]) -> str:
    """Build an EEGLAB-style key/value history command."""
    parts = []
    for key, value in options.items():
        if value is None:
            continue
        if isinstance(value, bool) and value is False:
            continue
        history_value = int(value) if isinstance(value, bool) else value
        parts.extend([format_history_value(key), format_history_value(history_value)])
    if not parts:
        return f"EEG = {function_name}(EEG);"
    return f"EEG = {function_name}(EEG, {', '.join(parts)});"


def value_history_command(function_name: str, values: list[Any], *, assignment: str) -> str:
    """Build an EEGLAB-style function call for scalar/helper pop functions."""
    formatted = ", ".join(format_history_value(value) for value in values)
    return f"{assignment} = {function_name}({formatted});"


def channel_controls(EEG: dict[str, Any]) -> tuple[ControlSpec, ...]:
    """Return the EEGLAB channel type/label selector rows used by pop_eegfiltnew."""
    labels = _channel_field_values(EEG, "labels")
    types = _channel_field_values(EEG, "type", unique=True)
    return (
        ControlSpec("text", "Channel type(s)"),
        ControlSpec("edit", tag="chantype", value=""),
        ControlSpec(
            "pushbutton",
            "...",
            tag="chantype_button",
            enabled=bool(types),
            callback=CallbackSpec(
                "select_channels",
                params={"button": "chantype_button", "target": "chantype", "channels": types},
                matlab_callback="pop_chansel({tmpchanlocs.type}, 'withindex', 'off')",
            ),
        ),
        ControlSpec("text", "OR channel labels or indices"),
        ControlSpec("edit", tag="channels", value=""),
        ControlSpec(
            "pushbutton",
            "...",
            tag="channels_button",
            enabled=bool(labels),
            callback=CallbackSpec(
                "select_channels",
                params={"button": "channels_button", "target": "channels", "channels": labels},
                matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
            ),
        ),
    )


def _channel_field_values(EEG: dict[str, Any], field: str, *, unique: bool = False) -> list[str]:
    values = [str(chan.get(field, "")) for chan in _chanlocs(EEG)]
    if not unique:
        return values
    deduped = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _chanlocs(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    return [chan if isinstance(chan, dict) else {} for chan in chanlocs_as_list(EEG.get("chanlocs"))]

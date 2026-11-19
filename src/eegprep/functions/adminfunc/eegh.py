"""EEGLAB-style command history helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_hist import eeg_hist
from eegprep.functions.miscfunc.matlab_inputs import restore_dataset_container, scalar_index, struct_records


def eegh(
    command: Any = None,
    history: list[str] | dict[str, Any] | list[dict[str, Any]] | None = None,
) -> Any:
    """Display or update EEGLAB-style command history.

    Args:
        command: Command string, numeric history operation, ``"find"``, or
            ``None`` to display history.
        history: Optional chronological ``ALLCOM`` list or EEG dataset whose
            ``history`` field should receive the command.

    Returns:
        The normalized command, rendered history text, or matched history item.
        Passing an EEG dataset returns that dataset. ``eeg_hist`` appends the
        command on its own line and leaves commands that contain ``pop_loadset``
        unrecorded.
    """
    command = _coerce_command(command)
    if isinstance(command, str) and command.strip().lower() == "find":
        return _find_history(history if _is_command_history(history) else [], None)
    if command is None:
        return _render_history(history if _is_command_history(history) else [])
    if isinstance(command, (int, float)) and not isinstance(command, bool):
        return _numeric_history(command, history if _is_command_history(history) else [])
    normalized = "" if command is None else str(command).strip()
    if not normalized:
        return ""
    records = _eeg_records(history)
    if records is not None:
        command_text = f"% multiple datasets command: {normalized}" if len(records) > 1 else normalized
        for eeg in records:
            _append_eeg_history(eeg, command_text)
        if isinstance(history, np.ndarray):
            return restore_dataset_container(history, records)
        return history
    if isinstance(history, list):
        _append_history(history, normalized)
    elif isinstance(history, dict):
        _append_eeg_history(history, normalized)
        return history
    return normalized


def _coerce_command(command: Any) -> Any:
    if isinstance(command, np.ndarray) and command.size == 0:
        return None
    index = scalar_index(command)
    if index is not None and isinstance(command, np.ndarray):
        return index
    return command


def eegh_find(history: list[str], text: str) -> str:
    """Return the most recent history item containing ``text``."""
    return _find_history(history, text)


def _append_history(history: list[str], command: str) -> None:
    if history and history[-1] == command:
        return
    history.append(command)


def _is_command_history(history: Any) -> bool:
    return isinstance(history, list) and not any(isinstance(item, dict) for item in history)


def _eeg_records(history: Any) -> list[dict[str, Any]] | None:
    if isinstance(history, dict):
        return None
    if isinstance(history, list) and history and all(isinstance(item, dict) for item in history):
        return history
    if isinstance(history, np.ndarray) and history.dtype.names and history.size:
        return struct_records(history)
    return None


def _append_eeg_history(eeg: dict[str, Any], command: str) -> None:
    eeg_hist(eeg, command)


def _render_history(history: list[str]) -> str:
    lines = [f"{index}. {command}" for index, command in enumerate(reversed(history), start=1)]
    return "\n".join(lines)


def _find_history(history: list[str], text: str | None) -> str:
    needle = "" if text is None else str(text)
    for command in reversed(history):
        if needle in command:
            return command
    return ""


def _numeric_history(command: int | float, history: list[str]) -> str:
    value = int(command)
    if float(command) != value:
        raise ValueError("eegh numeric command must be an integer")
    if value == 0:
        history.clear()
        return ""
    if value < 0:
        remove_count = min(abs(value), len(history))
        if remove_count:
            del history[-remove_count:]
        return ""
    if value > len(history):
        return ""
    return list(reversed(history))[value - 1]

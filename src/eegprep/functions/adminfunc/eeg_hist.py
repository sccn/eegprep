"""Append one command to an EEG dataset history."""

from __future__ import annotations

from typing import Any

import numpy as np


def eeg_hist(eeg: dict[str, Any], command: str) -> dict[str, Any]:
    """Append ``command`` to ``eeg["history"]`` and return the same dataset.

    The command is stored on its own line, including a leading newline when
    the history was empty. A character matrix or a list of rows is joined
    with newlines before that command is added. A command that contains
    ``pop_loadset`` is not recorded. A missing history field becomes an
    empty string.

    Args:
        eeg: EEG dataset dictionary.
        command: History command text.

    Returns:
        ``eeg``, updated in place.
    """
    if eeg.get("history") is None:
        eeg["history"] = ""
    if command and "pop_loadset" not in command:
        eeg["history"] = f"{_history_text(eeg['history'])}\n{command}"
    return eeg


def _history_text(history: Any) -> str:
    if isinstance(history, str):
        return history
    if isinstance(history, bytes):
        return history.decode()
    if isinstance(history, np.ndarray):
        return _array_history_text(history)
    if isinstance(history, (list, tuple)):
        return "\n".join(_history_text(item) for item in history)
    return "" if history is None else str(history)


def _array_history_text(history: np.ndarray) -> str:
    if history.size == 0:
        return ""
    if history.dtype == object:
        return "\n".join(_history_text(item) for item in history.reshape(-1))
    if history.dtype.kind not in "SU":
        return "\n".join(str(item) for item in history.reshape(-1))
    if history.ndim == 0:
        return _cell_text(history.item())
    if history.ndim == 2 and _is_character_matrix(history):
        return "\n".join("".join(_cell_text(cell) for cell in row) for row in history)
    return "\n".join(_cell_text(item) for item in history.reshape(-1))


def _is_character_matrix(history: np.ndarray) -> bool:
    if history.dtype.kind == "U":
        return history.dtype.itemsize // 4 == 1
    return history.dtype.itemsize == 1


def _cell_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)

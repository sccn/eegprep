"""Append one command to an EEG dataset history."""

from __future__ import annotations

from typing import Any


def eeg_hist(eeg: dict[str, Any], command: str) -> dict[str, Any]:
    """Append ``command`` to ``eeg["history"]`` and return the same dataset.

    The command is stored on its own line, including a leading newline when
    the history was empty. A command that contains ``pop_loadset`` is not
    recorded. A missing history field becomes an empty string.

    Args:
        eeg: EEG dataset dictionary.
        command: History command text.

    Returns:
        ``eeg``, updated in place.
    """
    history = eeg.get("history")
    if history is None or getattr(history, "size", None) == 0:
        history = ""
    elif not isinstance(history, str):
        history = str(history)
    if command and "pop_loadset" not in command:
        history = f"{history}\n{command}"
    eeg["history"] = history
    return eeg

"""EEGLAB-style command history helper."""

from __future__ import annotations


def eegh(command: str | None = None, history: list[str] | None = None) -> str:
    """Append a command to a session history list and return it.

    Args:
        command: EEGLAB-style command string. Empty commands are ignored.
        history: Optional list receiving the command.

    Returns:
        The normalized command string.
    """
    normalized = "" if command is None else str(command).strip()
    if normalized and history is not None:
        history.append(normalized)
    return normalized

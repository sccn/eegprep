"""EEGLAB-style command history helper."""

from __future__ import annotations

from typing import Any


def eegh(command: Any = None, history: list[str] | dict[str, Any] | None = None) -> str:
    """Display or update EEGLAB-style command history.

    Args:
        command: Command string, numeric history operation, ``"find"``, or
            ``None`` to display history.
        history: Optional chronological ``ALLCOM`` list or EEG dataset whose
            ``history`` field should receive the command.

    Returns:
        The normalized command, rendered history text, or matched history item.
    """
    if isinstance(command, str) and command.strip().lower() == "find":
        return _find_history(history if isinstance(history, list) else [], None)
    if command is None:
        return _render_history(history if isinstance(history, list) else [])
    if isinstance(command, (int, float)) and not isinstance(command, bool):
        return _numeric_history(command, history if isinstance(history, list) else [])
    normalized = "" if command is None else str(command).strip()
    if not normalized:
        return ""
    if isinstance(history, list):
        _append_history(history, normalized)
    elif isinstance(history, dict):
        _append_eeg_history(history, normalized)
    return normalized


def eegh_find(history: list[str], text: str) -> str:
    """Return the most recent history item containing ``text``."""
    return _find_history(history, text)


def _append_history(history: list[str], command: str) -> None:
    if history and history[-1] == command:
        return
    history.append(command)


def _append_eeg_history(eeg: dict[str, Any], command: str) -> None:
    existing = str(eeg.get("history") or "")
    line = command.rstrip(";") + ";"
    if existing.rstrip().splitlines()[-1:] == [line]:
        return
    eeg["history"] = (existing.rstrip() + "\n" + line).lstrip()


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

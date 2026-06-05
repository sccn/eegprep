"""EEGLAB-style wrapper for writing channel-location files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.sigprocfunc.writelocs import writelocs


def pop_writelocs(
    chans: Any,
    filename: str | Path | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> str:
    """Write channel locations and return a replayable history command."""
    if filename in (None, ""):
        return ""
    writelocs(chans, filename, *args, **kwargs)
    command = _history_command(filename, parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True))
    return command if return_com else command


def _history_command(filename: str | Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(Path(filename))]
    for key, value in options.items():
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"pop_writelocs(EEG['chanlocs'], {', '.join(pieces)});"


__all__ = ["pop_writelocs"]

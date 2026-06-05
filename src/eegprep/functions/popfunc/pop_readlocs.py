"""EEGLAB-style wrapper for reading channel-location files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args
from eegprep.functions.sigprocfunc.readlocs import readlocs


def pop_readlocs(
    filename: str | Path | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """Read channel locations and return an EEGPrep ``chanlocs`` list."""
    if filename in (None, ""):
        return ([], "") if return_com else []
    locs = readlocs(filename, *args, **kwargs)
    command = _history_command(filename, parse_key_value_args(args, kwargs, lowercase_keys=True, lowercase_kwargs=True))
    return (locs, command) if return_com else locs


def _history_command(filename: str | Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(Path(filename))]
    for key, value in options.items():
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG['chanlocs'] = pop_readlocs({', '.join(pieces)});"


__all__ = ["pop_readlocs"]

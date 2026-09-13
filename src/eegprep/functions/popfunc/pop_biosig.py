"""Import EEG data using BIOSIG-equivalent Python readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_fileio import pop_fileio


_BIOSIG_SUFFIXES = {".edf", ".bdf", ".gdf"}


def pop_biosig(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import BIOSIG-style EDF/BDF/GDF files, optionally over a time range."""
    path = Path(filename)
    if path.suffix.lower() not in _BIOSIG_SUFFIXES:
        raise ValueError(
            "pop_biosig supports EDF, BDF, and GDF files. Use pop_fileio or pop_loadset for other formats."
        )
    eeg, _command = pop_fileio(filename, return_com=True, **kwargs)
    arguments = [format_history_value(path)]
    for key, value in kwargs.items():
        arguments.extend([format_history_value(key), format_history_value(value)])
    command = f"EEG = pop_biosig({', '.join(arguments)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg

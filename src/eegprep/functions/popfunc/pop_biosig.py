"""Import EEG data using BIOSIG-equivalent Python readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc.pop_fileio import pop_fileio


def pop_biosig(filename: str | Path, *, return_com: bool = False, **kwargs: Any) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import BIOSIG-style EDF/BDF/GDF files."""
    eeg, _command = pop_fileio(filename, return_com=True, **kwargs)
    command = f"EEG = pop_biosig('{Path(filename)}');"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg

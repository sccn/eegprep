"""Save EEGPrep/EEGLAB-style command history scripts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import cell_items, is_cell
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_saveh(
    allcoms: str | list[str] | np.ndarray,
    filename: str | Path = "eegprephist.m",
    filepath: str | Path = ".",
) -> str:
    """Save dataset or session command history to a script file.

    A MATLAB cell or character array of commands is written one command per
    line, the same way as a Python list.
    """
    path = Path(filepath) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    commands = _history_commands(allcoms)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"% EEGPrep history file generated on {date.today().isoformat()}\n")
        stream.write("% ------------------------------------------------\n")
        if commands is not None:
            for command in commands:
                stream.write(f"{command}\n")
            stream.write("eegprep.eeglab();\n")
        else:
            text = str(allcoms)
            stream.write(text)
            if text and not text.endswith("\n"):
                stream.write("\n")
    return f"pop_saveh(ALLCOM, {format_history_value(path.name)}, {format_history_value(path.parent)});"


def _history_commands(allcoms: Any) -> list[str] | None:
    if is_cell(allcoms):
        return [str(item) for item in cell_items(allcoms)]
    if isinstance(allcoms, np.ndarray):
        if allcoms.size == 0:
            return []
        if allcoms.dtype.kind in "SU":
            return [str(item) for item in allcoms.reshape(-1)]
        return None
    if isinstance(allcoms, list):
        return [str(item) for item in allcoms]
    return None

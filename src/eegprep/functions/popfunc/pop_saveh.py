"""Save EEGPrep/EEGLAB-style command history scripts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.miscfunc.matlab_inputs import cell_items, is_cell
from eegprep.functions.popfunc._pop_utils import format_history_value

SESSION_REDRAW = "eegprep.eeglab();"
PYTHON_IMPORTS = ["import eegprep", "from eegprep import *"]
PYTHON_EMPTY_WORKSPACE = "ALLEEG, EEG, CURRENTSET, ALLCOM = [], {}, 0, []"


def pop_saveh(
    allcoms: str | list[str] | np.ndarray,
    filename: str | Path = "eegprephist.m",
    filepath: str | Path = ".",
) -> str:
    """Save dataset or session command history to a script file.

    A list, MATLAB cell, or character array is session history (``ALLCOM``),
    written one command per line in order and followed by a GUI redraw. A
    string is dataset history (``EEG.history``). A ``.py`` filename translates
    each command to Python; other suffixes keep the recorded MATLAB syntax.
    """
    path = Path(filepath) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    session_commands = _history_commands(allcoms)
    is_session = session_commands is not None
    commands = [*session_commands, SESSION_REDRAW] if is_session else str(allcoms).splitlines()
    is_python = path.suffix.lower() == ".py"
    comment = "#" if is_python else "%"
    lines = [
        f"{comment} EEGPrep history file generated on {date.today().isoformat()}",
        f"{comment} ------------------------------------------------",
    ]
    if is_python:
        lines += PYTHON_IMPORTS
        if is_session:
            lines.append(PYTHON_EMPTY_WORKSPACE)
        lines += [_python_line(command) for command in commands]
    else:
        lines += commands
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    target = "ALLCOM" if is_session else "EEG.history"
    return f"pop_saveh({target}, {format_history_value(path.name)}, {format_history_value(str(path.parent))});"


def _python_line(command: str) -> str:
    text = command.strip()
    if text.startswith("%"):
        return "#" + text[1:]
    return _console_python_command(text)


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

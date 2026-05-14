"""Save EEGPrep/EEGLAB-style command history scripts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


def pop_saveh(allcoms: str | list[str], filename: str | Path = "eegprephist.m", filepath: str | Path = ".") -> str:
    """Save dataset or session command history to a script file."""
    path = Path(filepath) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"% EEGPrep history file generated on {date.today().isoformat()}\n")
        stream.write("% ------------------------------------------------\n")
        if isinstance(allcoms, list):
            for command in reversed(allcoms):
                stream.write(f"{command}\n")
            stream.write("eegprep.eeglab();\n")
        else:
            stream.write(str(allcoms))
            if allcoms and not str(allcoms).endswith("\n"):
                stream.write("\n")
    return f"pop_saveh(ALLCOM, '{path.name}', '{path.parent}');"

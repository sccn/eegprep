"""Run EEGPrep history scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_runscript(filename: str | Path, namespace: dict[str, Any] | None = None) -> str:
    """Run a Python history script selected by the user.

    The selected Python file is executed in-process; MATLAB ``.m`` scripts are
    not run by EEGPrep.
    """
    path = Path(filename)
    if path.suffix.lower() == ".py":
        exec_globals = {} if namespace is None else namespace
        exec(path.read_text(encoding="utf-8"), exec_globals)
    elif path.suffix.lower() in {".m", ".txt"}:
        raise NotImplementedError("EEGPrep can only run Python history scripts; MATLAB/text scripts are not supported")
    else:
        raise ValueError("History scripts must be .py, .m, or .txt files")
    return f"LASTCOM = pop_runscript({format_history_value(path)});"

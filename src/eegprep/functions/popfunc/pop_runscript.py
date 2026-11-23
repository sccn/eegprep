"""Run EEGPrep history scripts."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from eegprep.functions.adminfunc.console import _console_python_command
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_runscript(filename: str | Path, namespace: dict[str, Any] | None = None) -> str:
    """Run a ``.py``, ``.m``, or ``.txt`` history script.

    Python files are executed as written. MATLAB and text history is translated
    with the same converter the console uses. A line that is still not Python
    after that translation is named in the error.
    """
    path = Path(filename)
    exec_globals = {} if namespace is None else namespace
    suffix = path.suffix.lower()
    if suffix == ".py":
        exec(path.read_text(encoding="utf-8"), exec_globals)
    elif suffix in {".m", ".txt"}:
        _run_history_lines(path, exec_globals)
    else:
        raise ValueError(
            f"pop_runscript: {path.name} is not a .py, .m, or .txt history script. "
            "Save the history as one of those files with File > Save history, then run it again."
        )
    return f"LASTCOM = pop_runscript({format_history_value(path)});"


def _run_history_lines(path: Path, namespace: dict[str, Any]) -> None:
    unsupported: list[int] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        python = _console_python_command(line)
        if not python.strip():
            continue
        try:
            ast.parse(python)
        except SyntaxError:
            unsupported.append(number)
            continue
        exec(python, namespace)
    if unsupported:
        shown = ", ".join(str(number) for number in unsupported)
        raise ValueError(
            f"pop_runscript: {path.name} line {shown} is not a supported pop_ or eeg_ history command. "
            "Remove that line or rewrite it as a Python call, then run the script again."
        )

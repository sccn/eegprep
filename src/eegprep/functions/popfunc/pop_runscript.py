"""Run EEGPrep history scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pop_runscript(filename: str | Path, namespace: dict[str, Any] | None = None) -> str:
    """Run a Python history script or record a MATLAB-style history script."""
    path = Path(filename)
    if path.suffix.lower() == ".py":
        exec_globals = {} if namespace is None else namespace
        exec(path.read_text(encoding="utf-8"), exec_globals)
    elif path.suffix.lower() not in {".m", ".txt"}:
        raise ValueError("History scripts must be .py, .m, or .txt files")
    return f"LASTCOM = pop_runscript('{path}');"

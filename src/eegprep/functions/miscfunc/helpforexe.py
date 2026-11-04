"""Write display scripts for function help text."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.vararg2str import vararg2str


def helpforexe(functions: str | Sequence[Any] | np.ndarray, folder: str | Path) -> None:
    """Write one help script per function into ``folder``.

    Each name may end in ``.m``. The help text is that EEGPrep function's
    docstring, which is the text EEGLAB would take from the leading ``%``
    block of the ``.m`` file. The output file is named ``help_<name>`` and
    contains one ``disp`` statement per docstring line.

    The generated scripts are not executed. EEGLAB changes directory and
    evaluates them to confirm they parse as MATLAB; that check belongs to
    the compiled MATLAB application.

    Args:
        functions: A function name, or several names in a list or cell array.
        folder: Directory that receives the help scripts. It must already exist.

    Raises:
        FileNotFoundError: A name is not an EEGPrep function.
    """
    destination = Path(folder)
    for name in _function_names(functions):
        text = inspect.getdoc(_function_for_help(name)) or ""
        (destination / f"help_{name}").write_text(_disp_script(text), encoding="utf-8")


def _function_names(functions: str | Sequence[Any] | np.ndarray) -> list[str]:
    if isinstance(functions, str):
        return [functions]
    if isinstance(functions, np.ndarray):
        return [str(item) for item in functions.ravel(order="F")]
    return [str(item) for item in functions]


def _function_for_help(name: str) -> Any:
    stem = name[:-2] if name.endswith(".m") else name
    function = getattr(importlib.import_module("eegprep"), stem, None)
    if not callable(function):
        raise FileNotFoundError(name)
    return function


def _disp_script(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip() == "":
            lines.append("disp(' ');\n")
        else:
            lines.append(f"disp({vararg2str(line)});\n")
    return "".join(lines)

"""Export ICA weights or inverse weights to a text file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_expica(EEG: dict[str, Any], whichica: str = "weights", filename: str | Path | None = None) -> str:
    """Export the ICA weight matrix or inverse weight matrix.

    Argument order matches ``pop_expica.m``: ``whichica`` is ``'weights'`` or
    ``'inv'``, then the output path. ``pop_expica.m`` line 79 writes the matrix
    with ``save -ascii``. Each value is ``%16.7e`` and values in a row are
    concatenated, which is the width MATLAB uses for that save flag.
    """
    if filename is None:
        raise ValueError("pop_expica: a filename is required")
    matrix = str(whichica).lower()
    if matrix in {"weights", "weight"}:
        values = finite_matmul(np.asarray(EEG.get("icaweights", [])), np.asarray(EEG.get("icasphere", [])))
    elif matrix in {"inv", "inverse", "icawinv"}:
        values = np.asarray(EEG.get("icawinv", []))
    else:
        raise ValueError(
            "pop_expica: matrix must be 'weights' or 'inv'. "
            "Pass 'weights' for the unmixing matrix or 'inv' for the inverse."
        )
    if values.size == 0:
        raise ValueError("pop_expica: No ICA matrix is available to export. Run ICA, then call pop_expica again.")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.atleast_2d(np.asarray(values, dtype=float))
    text = "\n".join("".join(f"{float(value):16.7e}" for value in row) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")
    return f"LASTCOM = pop_expica(EEG, {format_history_value(matrix)}, {format_history_value(path)});"

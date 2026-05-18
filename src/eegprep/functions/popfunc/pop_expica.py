"""Export ICA weights or inverse weights to a text file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_expica(EEG: dict[str, Any], filename: str | Path, matrix: str = "weights") -> str:
    """Export the ICA weight matrix or inverse weight matrix."""
    matrix = str(matrix).lower()
    if matrix in {"weights", "weight"}:
        values = np.asarray(EEG.get("icaweights", [])) @ np.asarray(EEG.get("icasphere", []))
    elif matrix in {"inv", "inverse", "icawinv"}:
        values = np.asarray(EEG.get("icawinv", []))
    else:
        raise ValueError("matrix must be 'weights' or 'inv'")
    if values.size == 0:
        raise ValueError("No ICA matrix is available to export")
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, values, delimiter="\t")
    return (
        "LASTCOM = pop_expica(EEG, "
        f"{format_history_value(path)}, {format_history_value(matrix)});"
    )

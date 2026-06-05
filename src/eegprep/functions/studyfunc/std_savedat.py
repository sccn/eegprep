"""Save EEGPrep STUDY measure data sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from eegprep.functions.popfunc._file_io import json_safe


def std_savedat(tmpfile: str | Path, structure: Any) -> Path:
    """Save a measure structure as JSON or MATLAB-compatible MAT data."""
    path = Path(tmpfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(json_safe(structure), indent=2, sort_keys=True), encoding="utf-8")
        return path
    savemat(path, _mat_payload(structure), do_compression=True)
    return path


def _mat_payload(structure: Any) -> dict[str, Any]:
    if isinstance(structure, dict):
        return {str(key): _mat_value(value) for key, value in structure.items()}
    return {"data": _mat_value(structure)}


def _mat_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except ValueError:
            return np.asarray(value, dtype=object)
    return value


__all__ = ["std_savedat"]

"""Write EEGLAB four-byte float files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def floatwrite(data: Any, filename: str | Path, format: str | None = None, transp: str = "normal") -> None:
    """Write a matrix as float32 values using MATLAB/EEGLAB ordering."""
    array = np.asarray(data, dtype=_dtype(format))
    if str(transp).lower() != "normal":
        array = array.T
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.ravel(array, order="F").tofile(filename)


def _dtype(format: str | None) -> np.dtype:
    text = str(format or "native").lower()
    if text in {"native", "n", ""}:
        return np.dtype("f4")
    if text in {"ieee-le", "l", "little", "float32le"}:
        return np.dtype("<f4")
    if text in {"ieee-be", "b", "big", "float32be"}:
        return np.dtype(">f4")
    raise ValueError(f"Unsupported float format: {format}")


__all__ = ["floatwrite"]

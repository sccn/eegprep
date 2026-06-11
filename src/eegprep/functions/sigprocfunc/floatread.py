"""Read EEGLAB four-byte float files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def floatread(filename: str | Path, size: Any, format: str | None = None, offset: Any = 0) -> np.ndarray:
    """Read a binary float32 matrix using MATLAB/EEGLAB column ordering."""
    dtype = _dtype(format)
    path = Path(filename)
    start = _offset(offset)
    values = np.fromfile(path, dtype=dtype, offset=start * np.dtype(dtype).itemsize)
    shape = _shape(size, values.size)
    count = int(np.prod(shape))
    if count > values.size:
        raise ValueError("Requested float matrix is larger than the file")
    return values[:count].reshape(shape, order="F")


def _dtype(format: str | None) -> np.dtype:
    text = str(format or "native").lower()
    if text in {"native", "n", ""}:
        return np.dtype("f4")
    if text in {"ieee-le", "l", "little", "float32le"}:
        return np.dtype("<f4")
    if text in {"ieee-be", "b", "big", "float32be"}:
        return np.dtype(">f4")
    raise ValueError(f"Unsupported float format: {format}")


def _offset(offset: Any) -> int:
    if isinstance(offset, (list, tuple)) and len(offset) == 2:
        dims = [int(value) for value in offset[0]]
        starts = [int(value) for value in offset[1]]
        if len(dims) != len(starts):
            raise ValueError("offset cell must contain matching dimensions and starts")
        flat = 0
        stride = 1
        for dim, start in zip(dims, starts):
            if start < 1 or start > dim:
                raise ValueError("offset starts must be 1-based and within data dimensions")
            flat += stride * (start - 1)
            stride *= dim
        return flat
    return int(offset or 0)


def _shape(size: Any, available: int) -> tuple[int, ...]:
    if isinstance(size, str):
        if size.lower() != "square":
            raise ValueError("floatread size string must be 'square'")
        side = int(np.sqrt(available))
        if side * side != available:
            raise ValueError("float file length is not square")
        return side, side
    dims = [float(value) for value in np.asarray(size, dtype=float).reshape(-1).tolist()]
    if not dims:
        raise ValueError("size must not be empty")
    if np.isinf(dims[-1]):
        known = int(np.prod(dims[:-1])) if len(dims) > 1 else 1
        if available % known:
            raise ValueError("float file length is not divisible by requested dimensions")
        dims[-1] = available // known
    return tuple(int(value) for value in dims)


__all__ = ["floatread"]

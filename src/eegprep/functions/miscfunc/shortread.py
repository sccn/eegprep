"""Read MATLAB-ordered int16 matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def shortread(
    filename: str | Path,
    size: Any,
    format: str | None = None,
    offset: int = 0,
) -> np.ndarray:
    """Read an int16 matrix using MATLAB column ordering.

    ``offset`` counts int16 values, not bytes. An infinite final dimension is
    inferred from the number of values remaining after the offset.
    """
    dtype = _dtype(format)
    start = _nonnegative_integer(offset)
    values = np.fromfile(Path(filename), dtype=dtype, offset=start * dtype.itemsize)
    shape = _shape(size, values.size)
    count = int(np.prod(shape))
    if count > values.size:
        raise ValueError("shortread: requested matrix is larger than the remaining file")
    return values[:count].reshape(shape, order="F")


def _dtype(format: str | None) -> np.dtype:
    text = str(format or "native").lower()
    if text in {"", "n", "native"}:
        return np.dtype("i2")
    if text in {"ieee-le", "l", "little", "int16le"}:
        return np.dtype("<i2")
    if text in {"ieee-be", "b", "big", "int16be"}:
        return np.dtype(">i2")
    raise ValueError(f"shortread: unsupported format {format!r}")


def _nonnegative_integer(value: Any) -> int:
    result = int(value)
    if result != value or result < 0:
        raise ValueError("shortread: offset must be a non-negative integer")
    return result


def _shape(size: Any, available: int) -> tuple[int, ...]:
    dimensions = np.asarray(size, dtype=float).reshape(-1)
    if dimensions.size == 0 or np.isnan(dimensions).any() or np.isinf(dimensions[:-1]).any():
        raise ValueError("shortread: only the final size dimension may be infinite")
    if np.any(dimensions[:-1] < 1) or np.any(dimensions[:-1] != np.floor(dimensions[:-1])):
        raise ValueError("shortread: finite size dimensions must be positive integers")
    if np.isneginf(dimensions[-1]):
        raise ValueError("shortread: the inferred final size dimension must be positive infinity")
    if np.isposinf(dimensions[-1]):
        known = int(np.prod(dimensions[:-1])) if dimensions.size > 1 else 1
        if available % known:
            raise ValueError("shortread: remaining file length is not divisible by the requested dimensions")
        dimensions[-1] = available // known
    if dimensions[-1] < 1 or dimensions[-1] != np.floor(dimensions[-1]):
        raise ValueError("shortread: finite size dimensions must be positive integers")
    return tuple(int(value) for value in dimensions)


__all__ = ["shortread"]

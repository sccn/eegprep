"""String selection helper for heterogeneous cell-like sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import cell_items, is_cell


def unique_cell_string(values: Iterable[Any]) -> list[str] | np.ndarray:
    """Return unique string elements in first-seen order, ignoring non-strings.

    A MATLAB cell is returned as a 1-by-N cell. A Python sequence is returned
    as a list.
    """
    cell = is_cell(values)
    if cell:
        values = cell_items(values)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    if cell:
        row = np.empty((1, len(result)), dtype=object)
        row[0, :] = result
        return row
    return result


__all__ = ["unique_cell_string"]

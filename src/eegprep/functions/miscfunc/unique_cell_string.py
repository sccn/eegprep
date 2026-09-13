"""String selection helper for heterogeneous cell-like sequences."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def unique_cell_string(values: Iterable[Any]) -> list[str]:
    """Return unique string elements in first-seen order, ignoring non-strings."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


__all__ = ["unique_cell_string"]

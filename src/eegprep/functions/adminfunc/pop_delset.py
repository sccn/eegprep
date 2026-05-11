"""Delete datasets from an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from typing import Any


def pop_delset(
    ALLEEG: list[dict[str, Any]] | None,
    indices: int | list[int] | tuple[int, ...],
) -> tuple[list[dict[str, Any]], str]:
    """Delete dataset indices from ``ALLEEG`` and return a history command."""
    alleeg = [] if ALLEEG is None else list(ALLEEG)
    if isinstance(indices, int):
        delete_indices = [int(indices)]
    else:
        delete_indices = [int(index) for index in indices]
    for index in sorted(set(delete_indices), reverse=True):
        if index < 1:
            raise ValueError("EEGLAB dataset indices are 1-based")
        if index <= len(alleeg):
            del alleeg[index - 1]
    command = f"ALLEEG = pop_delset( ALLEEG, {delete_indices} );"
    return alleeg, command

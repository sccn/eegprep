"""Delete datasets from an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from typing import Any


def pop_delset(
    ALLEEG: list[dict[str, Any]] | None,
    indices: int | list[int] | tuple[int, ...],
) -> tuple[list[dict[str, Any]], str]:
    """Delete datasets from ``ALLEEG`` and return a history command.

    Like EEGLAB, each deleted slot is emptied in place (an empty slot is ``{}``) so
    the remaining datasets keep their numbers; trailing empty slots are dropped.
    """
    alleeg = [] if ALLEEG is None else list(ALLEEG)
    delete_indices = [int(indices)] if isinstance(indices, int) else [int(index) for index in indices]
    for index in set(delete_indices):
        if index < 1:
            raise ValueError("EEGLAB dataset indices are 1-based")
        if index > len(alleeg):
            raise IndexError(f"No dataset at EEGLAB index {index}")
        alleeg[index - 1] = {}
    while alleeg and not alleeg[-1]:
        alleeg.pop()
    command = f"ALLEEG = pop_delset( ALLEEG, {delete_indices} );"
    return alleeg, command

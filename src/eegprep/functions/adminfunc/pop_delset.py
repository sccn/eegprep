"""Delete datasets from an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_inputs import as_dataset_list, matlab_ints, restore_dataset_container


def pop_delset(
    ALLEEG: list[dict[str, Any]] | np.ndarray | None,
    indices: Any,
) -> tuple[Any, str]:
    """Delete datasets from ``ALLEEG`` and return a history command.

    Like EEGLAB, each deleted slot is emptied in place (an empty slot is ``{}``) so
    the remaining datasets keep their numbers; trailing empty slots are dropped.
    A structure array keeps its shape, including emptied slots. Indices may be
    whole-number doubles.
    """
    delete_indices = matlab_ints(indices)
    if delete_indices is None:
        raise ValueError(
            "pop_delset: dataset indices must be whole numbers starting at 1. "
            "Pass a loaded dataset number or a list of them."
        )
    alleeg = as_dataset_list(ALLEEG)
    for index in set(delete_indices):
        if index < 1:
            raise ValueError(
                f"pop_delset: dataset index {index} is not valid. "
                "Dataset numbers start at 1. Pass a loaded dataset number."
            )
        if index > len(alleeg):
            raise ValueError(
                f"pop_delset: dataset {index} is not loaded. "
                f"There are {len(alleeg)} dataset slots. Pass a number from 1 to {len(alleeg)}."
            )
        alleeg[index - 1] = {}
    structured = isinstance(ALLEEG, np.ndarray) and bool(ALLEEG.dtype.names)
    if not structured:
        while alleeg and not alleeg[-1]:
            alleeg.pop()
    command = f"ALLEEG = pop_delset( ALLEEG, {delete_indices} );"
    return restore_dataset_container(ALLEEG, alleeg), command

"""Copy EEG datasets inside an EEGLAB-style ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
from eegprep.functions.miscfunc.matlab_inputs import as_dataset_list, restore_dataset_container, scalar_index
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec


def pop_copyset(
    ALLEEG: list[dict[str, Any]],
    set_in: int,
    set_out: int | None = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Copy dataset ``set_in`` to ``set_out`` using EEGLAB-facing indices.

    ``ALLEEG`` may be a Python list or a MATLAB struct array of datasets.
    ``set_in`` and ``set_out`` may be whole-number floats.
    """
    datasets = as_dataset_list(ALLEEG)
    if not datasets:
        raise ValueError("Pop_copyset error: cannot copy single dataset mode")
    source_index = scalar_index(set_in)
    if source_index is None:
        source_index = int(set_in)
    if source_index == 0:
        raise ValueError("Pop_copyset error: cannot copy dataset")
    source = eeg_retrieve(datasets, source_index)[0]
    if not isinstance(source, dict):
        raise ValueError("Pop_copyset error: cannot copy multiple datasets")
    if source.get("data") is None:
        raise ValueError("Pop_copyset error: cannot copy empty dataset")
    if gui is None:
        gui = set_out is None
    if gui:
        result = _run_gui(int(set_in), renderer=renderer)
        if result is None:
            return (ALLEEG, {}, 0, "") if return_com else (ALLEEG, {}, 0)
        set_out = result
    if set_out is None:
        set_out = source_index + 1
    destination = scalar_index(set_out)
    set_out = destination if destination is not None else int(set_out)
    if set_out < 1:
        raise ValueError("Output dataset index must be 1-based")
    copied = eeg_checkset(deepcopy(source))
    copied["saved"] = "no"
    output = list(datasets)
    while len(output) < set_out:
        output.append({})
    output[set_out - 1] = copied
    stored = restore_dataset_container(ALLEEG, output)
    if isinstance(stored, np.ndarray) and stored.dtype.names:
        position = np.unravel_index(set_out - 1, stored.shape, order="F")
        eeg_out = {field: deepcopy(stored[field][position]) for field in stored.dtype.names}
    else:
        eeg_out = deepcopy(copied)
    current = np.float64(set_out) if isinstance(stored, np.ndarray) else set_out
    command = f"[ALLEEG EEG CURRENTSET] = pop_copyset( ALLEEG, {source_index}, {set_out});"
    return (stored, eeg_out, current, command) if return_com else (stored, eeg_out, current)


def pop_copyset_dialog_spec(set_in: int) -> DialogSpec:
    """Return the EEGLAB-like ``pop_copyset`` dialog spec."""
    return DialogSpec(
        title="Copy dataset -- pop_copyset()",
        function_name="pop_copyset",
        eeglab_source="functions/popfunc/pop_copyset.m",
        size=(385, 160),
        help_text="pophelp('pop_copyset')",
        geometry=((1,), (1,)),
        controls=(
            ControlSpec("text", "Index of the new dataset:"),
            ControlSpec("edit", tag="set_out", value=str(set_in + 1)),
        ),
    )


def _run_gui(set_in: int, *, renderer: Any | None = None) -> int | None:
    result = inputgui(pop_copyset_dialog_spec(set_in), renderer=renderer)
    if result is None:
        return None
    return int(float(str(result.get("set_out") or "").strip()))


__all__ = ["pop_copyset", "pop_copyset_dialog_spec"]

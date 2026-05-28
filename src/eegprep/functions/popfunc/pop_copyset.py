"""Copy EEG datasets inside an EEGLAB-style ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_retrieve import eeg_retrieve
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
    """Copy dataset ``set_in`` to ``set_out`` using EEGLAB-facing indices."""
    if not ALLEEG:
        raise ValueError("Pop_copyset error: cannot copy single dataset mode")
    if int(set_in) == 0:
        raise ValueError("Pop_copyset error: cannot copy dataset")
    source = eeg_retrieve(ALLEEG, int(set_in))[0]
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
        set_out = int(set_in) + 1
    set_out = int(set_out)
    if set_out < 1:
        raise ValueError("Output dataset index must be 1-based")
    copied = eeg_checkset(deepcopy(source))
    copied["saved"] = "no"
    output = list(ALLEEG)
    while len(output) < set_out:
        output.append({})
    output[set_out - 1] = copied
    command = f"[ALLEEG EEG CURRENTSET LASTCOM] = pop_copyset(ALLEEG, {int(set_in)}, {set_out});"
    return (output, deepcopy(copied), set_out, command) if return_com else (output, deepcopy(copied), set_out)


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

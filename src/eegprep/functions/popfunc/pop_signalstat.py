"""EEGLAB-style wrapper for channel/component signal statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    component_activations,
    component_map_data,
    history_command,
    numeric_vector,
)
from eegprep.functions.sigprocfunc.signalstat import signalstat


def pop_signalstat(
    EEG: dict[str, Any] | None = None,
    typeproc: int = 1,
    cnum: Any = None,
    percent: float = 5,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Compute and plot statistics for one channel or component."""
    if EEG is None:
        return (None, "") if return_com else None
    typeproc = int(typeproc)
    if gui is None:
        gui = cnum is None
    if gui:
        result = _run_gui(EEG, typeproc=typeproc, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        cnum = result["cnum"]
        percent = result["percent"]
    if cnum is None:
        cnum = 1
    index = _single_index(cnum)
    stat_chanlocs = EEG.get("chanlocs", [])
    if typeproc == 1:
        data = np.asarray(EEG.get("data"), dtype=float)
        if data.ndim == 3:
            data = data.reshape(data.shape[0], -1)
        if index < 1 or index > data.shape[0]:
            raise ValueError(f"channel number must be 1-based and within 1..{data.shape[0]}")
        values = data[index - 1, :]
        dlabel = None
        dlabel2 = f"Channel {index}"
        map_value = index
    else:
        acts = component_activations(EEG)
        if index < 1 or index > acts.shape[0]:
            raise ValueError(f"component number must be 1-based and within 1..{acts.shape[0]}")
        values = acts[index - 1, :, :].reshape(-1)
        dlabel = "Component Activity"
        dlabel2 = f"Component {index}"
        maps, stat_chanlocs = component_map_data(EEG)
        map_value = maps[:, index - 1]
    result = signalstat(values, 1, dlabel, float(percent), dlabel2, map_value, stat_chanlocs)
    command = history_command("pop_signalstat", typeproc, index, percent)
    return (result, command) if return_com else result


def pop_signalstat_dialog_spec(EEG: dict[str, Any], *, typeproc: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_signalstat``."""
    _ = EEG
    controls = (
        ControlSpec("text", "Channel number:" if int(typeproc) else "Component number:"),
        ControlSpec("edit", tag="cnum", value="1"),
        ControlSpec("text", "Trim percentage (each end):"),
        ControlSpec("edit", tag="percent", value="5"),
    )
    return DialogSpec(
        title="Plot signal statistics -- pop_signalstat()",
        controls=controls,
        geometry=((1, 1), (1, 1)),
        function_name="pop_signalstat",
        eeglab_source="functions/popfunc/pop_signalstat.m",
        help_text="pophelp('pop_signalstat')",
        size=(420, 150),
    )


def _run_gui(EEG: dict[str, Any], *, typeproc: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_signalstat_dialog_spec(EEG, typeproc=typeproc), renderer=renderer)
    if result is None:
        return None
    return {
        "cnum": _single_index(result.get("cnum", 1)),
        "percent": float(numeric_vector(result.get("percent", 5))[0]),
    }


def _single_index(value: Any) -> int:
    values = numeric_vector(value, dtype=int)
    if values.size != 1:
        raise ValueError("channel/component number must be a single integer")
    return int(values[0])


__all__ = ["pop_signalstat", "pop_signalstat_dialog_spec"]

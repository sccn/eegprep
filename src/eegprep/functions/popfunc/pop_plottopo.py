"""EEGLAB-style wrapper for plotting channel ERPs in an array."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import data_time_slice, history_command, numeric_vector
from eegprep.functions.sigprocfunc.plottopo import plottopo


def pop_plottopo(
    EEG: dict[str, Any] | None = None,
    chans: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel ERP traces in a rectangular/scalp-like array."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = chans is None and not kwargs
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        chans = result["chans"]
        kwargs.update(result["options"])
    data, times = data_time_slice(EEG, kwargs.pop("timerange", None))
    erp = np.nanmean(data, axis=2)
    figure = plottopo(
        erp,
        times=times,
        chanlocs=EEG.get("chanlocs", []),
        channels=chans,
        title=str(kwargs.pop("title", EEG.get("setname") or "Channel ERPs")),
        ydir=int(kwargs.pop("ydir", -1)),
    )
    command = history_command("pop_plottopo", chans, **kwargs)
    return (figure, command) if return_com else figure


def pop_plottopo_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_plottopo``."""
    return DialogSpec(
        title="Topographic ERP plot - pop_plottopo()",
        controls=(
            ControlSpec("text", "Channels to plot"),
            ControlSpec("edit", tag="chans", value=f"1:{int(EEG.get('nbchan', 0) or 0)}"),
            ControlSpec("text", "Plot title"),
            ControlSpec("edit", tag="title", value=str(EEG.get("setname") or "Channel ERPs")),
            ControlSpec("text", "Plot single trials"),
            ControlSpec("checkbox", "(set=yes)", tag="singletrials", value=False),
            ControlSpec("text", "Plot in rect. array"),
            ControlSpec("checkbox", "(set=yes)", tag="rect", value=False),
            ControlSpec("text", "Other plot options (see help)"),
            ControlSpec("edit", tag="options", value="'ydir', 1"),
        ),
        geometry=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
        function_name="pop_plottopo",
        eeglab_source="functions/popfunc/pop_plottopo.m",
        help_text="pophelp('pop_plottopo')",
        size=(524, 299),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_plottopo_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "chans": numeric_vector(result.get("chans", []), dtype=int).tolist(),
        "options": {
            "title": str(result.get("title", "") or ""),
            "singletrials": bool(result.get("singletrials", False)),
            "rect": bool(result.get("rect", False)),
            "options": str(result.get("options", "") or ""),
        },
    }


__all__ = ["pop_plottopo", "pop_plottopo_dialog_spec"]

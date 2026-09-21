"""EEGLAB-style wrapper for plotting channel ERPs in an array."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    data_time_slice,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.sigprocfunc.plottopo import plottopo


def pop_plottopo(
    EEG: dict[str, Any] | None = None,
    chans: Any = None,
    plottitle: str = "",
    singletrials: int = 0,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel ERP traces in a rectangular/scalp-like array.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = chans is None and not options
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        chans = result["chans"]
        plottitle = result["plottitle"]
        singletrials = int(result["singletrials"])
        options.update(result["options"])
    command_options = dict(options)
    data, times = data_time_slice(EEG, options.pop("timerange", None))
    rect = bool(options.pop("rect", False))
    plot_options = parse_plot_options_text(options.pop("options", ""))
    ydir = int(plot_options.pop("ydir", options.pop("ydir", -1)))
    title = str(options.pop("title", plottitle or EEG.get("setname") or "Channel ERPs"))
    plot_data = data if bool(singletrials) else np.nanmean(data, axis=2)
    figure = plottopo(
        plot_data,
        times=times,
        chanlocs=EEG.get("chanlocs", []),
        channels=chans,
        title=title,
        ydir=ydir,
        rect=rect,
        singletrials=bool(singletrials),
    )
    command = history_command("pop_plottopo", chans, plottitle, int(bool(singletrials)), **command_options)
    show_figures(figure, plot=plot)
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
        "plottitle": str(result.get("title", "") or ""),
        "singletrials": bool(result.get("singletrials", False)),
        "options": {
            "rect": bool(result.get("rect", False)),
            "options": str(result.get("options", "") or ""),
        },
    }


__all__ = ["pop_plottopo", "pop_plottopo_dialog_spec"]

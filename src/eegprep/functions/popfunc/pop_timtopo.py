"""EEGLAB-style wrapper for ERP traces with scalp maps."""

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
from eegprep.functions.sigprocfunc.timtopo import timtopo


def pop_timtopo(
    EEG: dict[str, Any] | None = None,
    plottimes: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel ERP traces and scalp maps at selected latencies."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = plottimes is None and not kwargs
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        plottimes = result["plottimes"]
        kwargs.update(result["options"])
    command_kwargs = dict(kwargs)
    data, times = data_time_slice(EEG, kwargs.pop("timerange", None))
    erp = np.nanmean(data, axis=2)
    winsize = _first_float(kwargs.pop("winsize", None), default=0.0)
    topoplot_options = parse_plot_options_text(kwargs.pop("options", ""))
    figure = timtopo(
        erp,
        EEG.get("chanlocs", []),
        times=times,
        plottimes=numeric_vector(plottimes).tolist(),
        winsize=winsize,
        title=str(kwargs.pop("title", EEG.get("setname") or "Channel ERPs")),
        topoplot_options=topoplot_options,
    )
    command = history_command("pop_timtopo", plottimes, **command_kwargs)
    show_figures(figure)
    return (figure, command) if return_com else figure


def pop_timtopo_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_timtopo``."""
    return DialogSpec(
        title="Channel ERPs with scalp maps -- pop_timtopo()",
        controls=(
            ControlSpec("text", "Plotting time range (ms):"),
            ControlSpec(
                "edit",
                tag="timerange",
                value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("text", "Scalp map latencies (ms, NaN -> max-RMS)"),
            ControlSpec("edit", tag="plottimes", value="NaN"),
            ControlSpec("text", "Window size at latencies in ms"),
            ControlSpec("edit", tag="winsize", value="0"),
            ControlSpec("text", "Plot title:"),
            ControlSpec(
                "edit", tag="title", value=f"ERP data and scalp maps of {str(EEG.get('setname') or '').strip()}".strip()
            ),
            ControlSpec("text", "Scalp map options (see >> help topoplot):"),
            ControlSpec("edit", tag="options", value=""),
        ),
        geometry=((2, 1), (2, 1), (2, 1), (2, 1), (2, 1)),
        function_name="pop_timtopo",
        eeglab_source="functions/popfunc/pop_timtopo.m",
        help_text="pophelp('pop_timtopo')",
        size=(597, 299),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_timtopo_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "plottimes": numeric_vector(result.get("plottimes", [])).tolist(),
        "options": {
            "timerange": numeric_vector(result.get("timerange", [])).tolist(),
            "title": str(result.get("title", "") or ""),
            "winsize": numeric_vector(result.get("winsize", [])).tolist(),
            "options": str(result.get("options", "") or ""),
        },
    }


def _first_float(value: Any, *, default: float) -> float:
    vector = numeric_vector(value)
    if vector.size == 0:
        return default
    return float(vector[0])


__all__ = ["pop_timtopo", "pop_timtopo_dialog_spec"]

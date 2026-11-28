"""EEGLAB-style wrapper for ERP traces with scalp maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
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
    timerange: Any = None,
    plottimes: Any = None,
    plottitle: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot channel ERP traces and scalp maps at selected latencies.

    Positional arguments follow EEGLAB: time range, scalp-map latencies, then
    the plot title. Pass ``plot='off'`` to build and return the figure without
    opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = timerange is None and plottimes is None and plottitle is None and not args and not kwargs
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        plottimes = result["plottimes"]
        kwargs.update(result["options"])
    command_kwargs = parse_key_value_args(args, kwargs, lowercase_kwargs=False) if args else dict(kwargs)
    all_options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    text_options = parse_plot_options_text(all_options.pop("options", ""))
    command_kwargs.pop("options", None)
    merged_options = {**text_options, **all_options}
    if timerange is None:
        timerange = merged_options.pop("timerange", None)
    else:
        merged_options.pop("timerange", None)
    command_kwargs.pop("timerange", None)
    if plottimes is None and "plottimes" in merged_options:
        plottimes = merged_options.pop("plottimes")
    else:
        merged_options.pop("plottimes", None)
    if plottitle is None:
        plot_title = str(merged_options.pop("title", EEG.get("setname") or "Channel ERPs"))
    else:
        merged_options.pop("title", None)
        plot_title = str(plottitle)
    command_kwargs.pop("title", None)
    winsize_value = merged_options.pop("winsize", None)
    command_kwargs.pop("winsize", None)
    limits_value = merged_options.pop("limits", None)
    voffsets_value = merged_options.pop("voffsets", None)
    plotchans_value = merged_options.pop("plotchans", None)
    plotenvelope_value = merged_options.pop("plotenvelope", 0)
    merged_options.pop("verbose", None)
    data, times = data_time_slice(EEG, timerange)
    erp = np.nanmean(data, axis=2)
    winsize = _first_float(winsize_value, default=0.0)
    figure = timtopo(
        erp,
        EEG.get("chanlocs", []),
        times=times,
        limits=limits_value,
        plottimes=numeric_vector(plottimes).tolist(),
        winsize=winsize,
        plotchans=plotchans_value,
        voffsets=voffsets_value,
        plotenvelope=plotenvelope_value,
        title=plot_title,
        topoplot_options=merged_options,
    )
    history_range = timerange
    if history_range is None or history_range == []:
        history_range = [float(times[0]), float(times[-1])] if len(times) else []
    if winsize:
        command_kwargs["winsize"] = winsize_value
    command = history_command(
        "pop_timtopo",
        history_range,
        numeric_vector(plottimes).tolist(),
        plot_title,
        leading_figure=True,
        **command_kwargs,
    )
    show_figures(figure, plot=plot)
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

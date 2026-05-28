"""EEGLAB-style wrapper for component ERP envelopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    component_maps,
    data_time_slice,
    eeg_times_ms,
    history_command,
    numeric_vector,
)
from eegprep.functions.sigprocfunc.envtopo import envtopo


def pop_envtopo(
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    timerange: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot largest component ERP envelopes and component maps."""
    if EEG is None:
        return (None, "") if return_com else None
    dataset = EEG[-1] if isinstance(EEG, list) else EEG
    if gui is None:
        gui = timerange is None and not kwargs
    if gui:
        result = _run_gui(dataset, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        timerange = result["timerange"]
        kwargs.update(result["options"])
    data, times = data_time_slice(dataset, timerange)
    icaweights = np.asarray(dataset.get("icaweights", []), dtype=float)
    icasphere = np.asarray(dataset.get("icasphere", []), dtype=float)
    if icaweights.size == 0 or icasphere.size == 0:
        raise ValueError("pop_envtopo requires ICA weights")
    weights = icaweights @ icasphere
    figure = envtopo(
        np.nanmean(data, axis=2),
        weights,
        times=times if times.size else eeg_times_ms(dataset),
        chanlocs=dataset.get("chanlocs", []),
        icawinv=component_maps(dataset),
        components=kwargs.pop("compnums", kwargs.pop("components", None)),
        title=str(kwargs.pop("title", dataset.get("setname") or "Largest ERP components")),
    )
    command = history_command("pop_envtopo", timerange, **kwargs)
    return (figure, command) if return_com else figure


def pop_envtopo_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_envtopo``."""
    return DialogSpec(
        title="Plot component and ERP envelopes -- pop_envtopo()",
        controls=(
            ControlSpec("text", "Enter time range (in ms) to plot:"),
            ControlSpec(
                "edit",
                tag="timerange",
                value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("text", "Enter time range (in ms) to rank component contributions:"),
            ControlSpec(
                "edit",
                tag="limcontrib",
                value=f"{float(EEG.get('xmin', 0)) * 1000:g} {float(EEG.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("text", "Number of largest contributing components to plot (7):"),
            ControlSpec("edit", tag="compsplot", value="7"),
            ControlSpec("text", "Else plot these component numbers only (Ex: 2:4,7):"),
            ControlSpec("edit", tag="components", value=""),
            ControlSpec("text", "Component numbers to remove from data before plotting:"),
            ControlSpec("edit", tag="subcomps", value=""),
            ControlSpec("text", "Plot title:"),
            ControlSpec("edit", tag="title", value=str(EEG.get("setname") or "Largest ERP components")),
            ControlSpec("text", "Optional topoplot() and envtopo() arguments:"),
            ControlSpec("edit", tag="options", value="'electrodes', 'off'"),
        ),
        geometry=((2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1), (2, 1)),
        function_name="pop_envtopo",
        eeglab_source="functions/popfunc/pop_envtopo.m",
        help_text="pophelp('pop_envtopo')",
        size=(825, 369),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_envtopo_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "timerange": numeric_vector(result.get("timerange", [])).tolist(),
        "options": {
            "components": numeric_vector(result.get("components", []), dtype=int).tolist(),
            "limcontrib": numeric_vector(result.get("limcontrib", [])).tolist(),
            "compsplot": numeric_vector(result.get("compsplot", []), dtype=int).tolist(),
            "subcomps": numeric_vector(result.get("subcomps", []), dtype=int).tolist(),
            "title": str(result.get("title", "") or ""),
            "options": str(result.get("options", "") or ""),
        },
    }


__all__ = ["pop_envtopo", "pop_envtopo_dialog_spec"]

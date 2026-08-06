"""EEGLAB-style wrapper for component ERP envelopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import (
    component_channel_indices,
    component_maps,
    data_time_slice,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.sigprocfunc.envtopo import envtopo


def pop_envtopo(
    EEG: dict[str, Any] | list[dict[str, Any]] | None = None,
    timerange: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot largest component ERP envelopes and component maps.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if isinstance(EEG, list):
        if len(EEG) != 1:
            raise ValueError(
                "pop_envtopo accepts one dataset in EEGPrep; multi-dataset envelope comparison is not a "
                "standalone runtime workflow."
            )
        dataset = EEG[0]
    else:
        dataset = EEG
    if gui is None:
        gui = timerange is None and not kwargs
    if gui:
        result = _run_gui(dataset, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        timerange = result["timerange"]
        kwargs.update(result["options"])
    command_kwargs = dict(kwargs)
    data, times = data_time_slice(dataset, timerange)
    icaweights = np.asarray(dataset.get("icaweights", []), dtype=float)
    icasphere = np.asarray(dataset.get("icasphere", []), dtype=float)
    if icaweights.size == 0 or icasphere.size == 0:
        raise ValueError("pop_envtopo requires ICA weights")
    icachansind = component_channel_indices(dataset, data.shape[0])
    data = data[icachansind, :, :]
    weights = icaweights @ icasphere
    if weights.shape[1] != data.shape[0]:
        raise ValueError("ICA weights do not match EEG.icachansind channel count")
    maps = component_maps(dataset)
    chanlocs = _component_chanlocs(dataset, maps, icachansind)
    components = kwargs.pop("compnums", kwargs.pop("components", None))
    max_components = _first_int(kwargs.pop("compsplot", None), default=7)
    title = str(kwargs.pop("title", dataset.get("setname") or "Largest ERP components"))
    topoplot_options = parse_plot_options_text(kwargs.pop("options", ""))
    figure = envtopo(
        np.nanmean(data, axis=2),
        weights,
        times=times if times.size else eeg_times_ms(dataset),
        chanlocs=chanlocs,
        icawinv=maps,
        components=components,
        max_components=max_components,
        rank_window=kwargs.pop("limcontrib", None),
        exclude_components=kwargs.pop("subcomps", None),
        topoplot_options=topoplot_options,
        title=title,
    )
    command = history_command("pop_envtopo", timerange, **command_kwargs)
    show_figures(figure, plot=plot)
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


def _component_chanlocs(EEG: dict[str, Any], maps: np.ndarray, icachansind: np.ndarray) -> Any:
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if maps.shape[0] == len(chanlocs):
        return chanlocs
    if maps.shape[0] == icachansind.size and chanlocs and np.max(icachansind) < len(chanlocs):
        return [chanlocs[index] for index in icachansind]
    raise ValueError("EEG.icawinv must have one row per channel location or ICA channel")


def _first_int(value: Any, *, default: int) -> int:
    vector = numeric_vector(value, dtype=int)
    if vector.size == 0:
        return default
    return int(vector[0])


__all__ = ["pop_envtopo", "pop_envtopo_dialog_spec"]

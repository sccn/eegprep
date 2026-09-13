"""EEGLAB-style wrapper for plotting component data arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    component_activations,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
    selected_indices,
    show_figures,
)
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.plottopo import plottopo


def pop_plotdata(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    indices: Any = None,
    trials: Any = None,
    plottitle: str = "",
    singletrials: int = 0,
    ydir: int = 1,
    ylimits: Any = None,
    *,
    components: Any = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    title: str | None = None,
):
    """Plot channel or component activity in a rectangular/scalp array.

    ``typeplot=1`` selects channels and ``typeplot=0`` selects ICA components.
    Trial selections and channel/component indices are EEGLAB-facing and
    therefore 1-based. When ``singletrials`` is false, selected trials are
    averaged before plotting; otherwise every selected trial is overlaid.

    ``components=...`` is the EEGPrep convenience spelling for
    ``typeplot=0, indices=...``. Pass ``plot='off'`` to build and return the
    figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if components is not None:
        if indices is not None:
            raise TypeError("indices and components cannot both be supplied")
        typeplot = 0
        indices = components
    typeplot = int(typeplot)
    if typeplot not in {0, 1}:
        raise ValueError("typeplot must be 1 for channels or 0 for components")
    if gui is None:
        gui = indices is None
    if gui:
        result = _run_gui(EEG, typeplot=typeplot, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        indices = result["indices"]
        plottitle = result["plottitle"]
        ylimits = result["ylimits"]
        singletrials = 0
        ydir = -1
    if title is not None:
        plottitle = title

    source = eeg_epoch_data(EEG) if typeplot else component_activations(EEG)
    row_indices = selected_indices(indices, source.shape[0])
    trial_indices = selected_indices(trials, source.shape[2])
    selected = source[row_indices, :, :][:, :, trial_indices]
    plot_data = selected if int(bool(singletrials)) else np.nanmean(selected, axis=2)
    plot_chanlocs = _selected_chanlocs(EEG, row_indices) if typeplot else _component_labels(row_indices)
    default_title = "Channel ERPs" if typeplot else "Component ERPs"
    title_value = str(plottitle or EEG.get("setname") or default_title)
    figure = plottopo(
        plot_data,
        times=eeg_times_ms(EEG),
        chanlocs=plot_chanlocs,
        title=title_value,
        ydir=int(ydir),
        ylimits=ylimits,
        rect=True,
        singletrials=bool(singletrials),
    )
    command = history_command(
        "pop_plotdata",
        typeplot,
        (row_indices + 1).tolist(),
        (trial_indices + 1).tolist(),
        plottitle,
        int(bool(singletrials)),
        int(ydir),
        numeric_vector(ylimits).tolist() or [0, 0],
    )
    show_figures(figure, plot=plot)
    return (figure, command) if return_com else figure


def pop_plotdata_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_plotdata``."""
    is_channel = bool(int(typeplot))
    count = int(EEG.get("nbchan", 0) or 0) if is_channel else np.asarray(EEG.get("icaweights", [])).shape[0]
    label = "Channel" if is_channel else "Component"
    return DialogSpec(
        title=f"{label} ERPs in rect. array -- pop_plotdata()",
        controls=(
            ControlSpec("text", f"{label} number(s):"),
            ControlSpec("edit", tag="indices", value=f"1:{count}" if count else ""),
            ControlSpec("text", "Plot title:"),
            ControlSpec("edit", tag="plottitle", value=f"{str(EEG.get('setname') or '').strip()} ERP".strip()),
            ControlSpec("text", "Vertical limits ([0 0]-> data range):"),
            ControlSpec("edit", tag="ylimits", value="0 0"),
        ),
        geometry=((1, 1), (1, 1), (1, 1)),
        function_name="pop_plotdata",
        eeglab_source="functions/popfunc/pop_plotdata.m",
        help_text="pophelp('pop_plotdata')",
        size=(490, 231),
    )


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_plotdata_dialog_spec(EEG, typeplot=typeplot), renderer=renderer)
    if result is None:
        return None
    return {
        "indices": numeric_vector(result.get("indices", []), dtype=int).tolist(),
        "plottitle": str(result.get("plottitle", "") or ""),
        "ylimits": numeric_vector(result.get("ylimits", [])).tolist(),
    }


def _selected_chanlocs(EEG: dict[str, Any], indices: np.ndarray) -> list[dict[str, Any]]:
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if len(chanlocs) < int(EEG.get("nbchan", 0) or 0):
        return [{"labels": str(index + 1)} for index in indices]
    return [chanlocs[int(index)] for index in indices]


def _component_labels(indices: np.ndarray) -> list[dict[str, str]]:
    return [{"labels": str(int(index) + 1)} for index in indices]


__all__ = ["pop_plotdata", "pop_plotdata_dialog_spec"]

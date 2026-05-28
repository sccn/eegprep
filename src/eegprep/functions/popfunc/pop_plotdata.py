"""EEGLAB-style wrapper for plotting component data arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import component_activations, eeg_times_ms, history_command, numeric_vector
from eegprep.functions.sigprocfunc.plottopo import plottopo


def pop_plotdata(
    EEG: dict[str, Any] | None = None,
    components: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot component ERP activations in a rectangular array."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = components is None and not kwargs
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        components = result["components"]
        kwargs.update(result["options"])
    acts = component_activations(EEG)
    erp = np.nanmean(acts, axis=2)
    command_kwargs = dict(kwargs)
    ylimits = kwargs.pop("ylimits", None)
    figure = plottopo(
        erp,
        times=eeg_times_ms(EEG),
        channels=components,
        title=str(kwargs.pop("title", EEG.get("setname") or "Component ERPs")),
        ydir=int(kwargs.pop("ydir", -1)),
        ylimits=ylimits,
    )
    command = history_command("pop_plotdata", components, **command_kwargs)
    return (figure, command) if return_com else figure


def pop_plotdata_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_plotdata``."""
    n_components = np.asarray(EEG.get("icaweights", [])).shape[0]
    return DialogSpec(
        title="Component ERPs in rect. array -- pop_plotdata()",
        controls=(
            ControlSpec("text", "Component number(s):"),
            ControlSpec("edit", tag="components", value=f"1:{n_components}" if n_components else ""),
            ControlSpec("text", "Plot title:"),
            ControlSpec("edit", tag="title", value=f"{str(EEG.get('setname') or '').strip()} ERP".strip()),
            ControlSpec("text", "Vertical limits ([0 0]-> data range):"),
            ControlSpec("edit", tag="ylimits", value="0 0"),
        ),
        geometry=((1, 1), (1, 1), (1, 1)),
        function_name="pop_plotdata",
        eeglab_source="functions/popfunc/pop_plotdata.m",
        help_text="pophelp('pop_plotdata')",
        size=(490, 231),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_plotdata_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "components": numeric_vector(result.get("components", []), dtype=int).tolist(),
        "options": {
            "title": str(result.get("title", "") or ""),
            "ylimits": numeric_vector(result.get("ylimits", [])).tolist(),
        },
    }


__all__ = ["pop_plotdata", "pop_plotdata_dialog_spec"]

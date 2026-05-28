"""EEGLAB-style wrapper for 3-D scalp maps."""

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
from eegprep.functions.sigprocfunc.headplot import headplot


def pop_headplot(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    items: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot ERP or component maps on a static 3-D head view."""
    if EEG is None:
        return ([], "") if return_com else []
    typeplot = int(typeplot)
    if gui is None:
        gui = items is None
    if gui:
        result = _run_gui(EEG, typeplot=typeplot, renderer=renderer)
        if result is None:
            return ([], "") if return_com else []
        items = result["items"]
        kwargs.update(result["options"])
    values, labels = _headplot_maps(EEG, typeplot, numeric_vector(items))
    figures = [
        headplot(map_values, EEG.get("chanlocs", []), title=str(kwargs.get("title") or label))
        for map_values, label in zip(values, labels)
    ]
    command = history_command("pop_headplot", typeplot, numeric_vector(items).tolist(), **kwargs)
    return (figures, command) if return_com else figures


def pop_headplot_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_headplot``."""
    label = "Latencies (ms)" if int(typeplot) else "Component indices"
    default_items = "0" if int(typeplot) else "1"
    return DialogSpec(
        title=("ERP" if int(typeplot) else "Component") + " map series in 3-D -- pop_headplot()",
        controls=(
            ControlSpec("text", f"{label} to plot:"),
            ControlSpec("edit", tag="items", value=default_items),
            ControlSpec("text", "Plot title:"),
            ControlSpec("edit", tag="title", value=str(EEG.get("setname") or "")),
        ),
        geometry=((2, 1), (2, 1)),
        function_name="pop_headplot",
        eeglab_source="functions/popfunc/pop_headplot.m",
        help_text="pophelp('pop_headplot')",
        size=(620, 230),
    )


def _run_gui(EEG: dict[str, Any], *, typeplot: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_headplot_dialog_spec(EEG, typeplot=typeplot), renderer=renderer)
    if result is None:
        return None
    return {
        "items": numeric_vector(result.get("items", [])).tolist(),
        "options": {"title": str(result.get("title", "") or "")},
    }


def _headplot_maps(EEG: dict[str, Any], typeplot: int, items: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
    if items.size == 0:
        raise ValueError("items must contain at least one latency or component index")
    if typeplot:
        data, _times = data_time_slice(EEG, None)
        erp = np.nanmean(data, axis=2)
        all_times = eeg_times_ms(EEG)
        maps = []
        labels = []
        for latency in items:
            if latency < np.nanmin(all_times) or latency > np.nanmax(all_times):
                raise ValueError("requested latency is outside the epoch time range")
            frame = int(np.argmin(np.abs(all_times - latency)))
            maps.append(erp[:, frame])
            labels.append(f"{latency:g} ms")
        return maps, labels
    icawinv = component_maps(EEG)
    maps = []
    labels = []
    for component in items.astype(int):
        if component < 1 or component > icawinv.shape[1]:
            raise ValueError("component index is outside available ICA components")
        maps.append(icawinv[:, component - 1])
        labels.append(f"IC {component}")
    return maps, labels


__all__ = ["pop_headplot", "pop_headplot_dialog_spec"]

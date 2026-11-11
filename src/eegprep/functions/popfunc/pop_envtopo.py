"""EEGLAB-style wrapper for component ERP envelopes."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.miscfunc.misc import finite_matmul
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
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Plot largest component ERP envelopes and component maps.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    datasets = list(EEG) if isinstance(EEG, list) else [EEG]
    if gui is None:
        gui = timerange is None and not kwargs
    subindices: tuple[int, int] | None = None
    if gui:
        compare_gui = len(datasets) > 1
        result = _run_gui(datasets if compare_gui else datasets[0], renderer=renderer, compare=compare_gui)
        if result is None:
            return (None, "") if return_com else None
        timerange = result["timerange"]
        kwargs.update(result["options"])
        if compare_gui:
            subindices = result["subindices"]
            datasets = [datasets[subindices[0] - 1], datasets[subindices[1] - 1]]
    if len(datasets) > 2:
        raise ValueError("pop_envtopo: Cannot process more than two datasets")
    compare = len(datasets) > 1
    if compare and subindices is None:
        subindices = (1, 2)
    if compare:
        dataset = datasets[0]
        other = datasets[1]
    else:
        dataset = datasets[0]
        other = None
    command_kwargs = dict(kwargs)
    prepared = _component_mean(dataset, timerange)
    if other is not None:
        other_prepared = _component_mean(other, timerange)
        if not np.array_equal(prepared["weights"], other_prepared["weights"]):
            raise ValueError("pop_envtopo: The ICA decomposition must be the same for the two datasets")
        if prepared["mean"].shape != other_prepared["mean"].shape:
            raise ValueError("pop_envtopo: the two datasets must have the same channels and time points")
        data_mean = prepared["mean"] - other_prepared["mean"]
    else:
        data_mean = prepared["mean"]
    times = prepared["times"]
    label_times = times if times.size else eeg_times_ms(dataset)
    topoplot_options = parse_plot_options_text(kwargs.pop("options", ""))
    compnums = kwargs.pop("compnums", None)
    if compnums is None:  # GUI supplies "components"; the console uses "compnums"
        compnums = kwargs.pop("components", None)
    title = kwargs.pop("title", None)
    if title is None or str(title).strip() == "":
        title = _envtopo_title(dataset)
    result = envtopo(
        data_mean,
        prepared["weights"],
        chanlocs=prepared["chanlocs"],
        icawinv=prepared["maps"],
        timerange=[float(label_times[0]), float(label_times[-1])],
        limcontrib=kwargs.pop("limcontrib", None),
        compnums=compnums,
        compsplot=_first_int(kwargs.pop("compsplot", None), default=7),
        subcomps=kwargs.pop("subcomps", 0),
        sortvar=str(kwargs.pop("sortvar", topoplot_options.pop("sortvar", "mp"))),
        envmode=str(kwargs.pop("envmode", topoplot_options.pop("envmode", "avg"))),
        sumenv=str(kwargs.pop("sumenv", topoplot_options.pop("sumenv", "fill"))),
        vert=kwargs.pop("vert", None),
        title=str(title),
        topoplot_options=topoplot_options,
    )
    history_range = timerange
    if history_range is None or history_range == []:
        history_range = [float(label_times[0]), float(label_times[-1])] if label_times.size else []
    eeg_name = "EEG" if subindices is None else f"EEG([{subindices[0]} {subindices[1]}])"
    command = history_command("pop_envtopo", history_range, eeg_name=eeg_name, **command_kwargs)
    show_figures(result.figure, plot=plot)
    return (result.figure, command) if return_com else result.figure


def pop_envtopo_dialog_spec(EEG: dict[str, Any] | list[dict[str, Any]], *, compare: bool = False) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_envtopo``."""
    dataset = EEG[-1] if isinstance(EEG, list) else EEG
    controls: list[ControlSpec] = []
    if compare:
        controls.extend(
            [
                ControlSpec("text", "Dataset indices to subtract (Ex: '1 2'-> 1-2)"),
                ControlSpec("edit", tag="subindices", value="2 1"),
            ]
        )
    controls.extend(
        [
            ControlSpec("text", "Enter time range (in ms) to plot:"),
            ControlSpec(
                "edit",
                tag="timerange",
                value=f"{float(dataset.get('xmin', 0)) * 1000:g} {float(dataset.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("text", "Enter time range (in ms) to rank component contributions:"),
            ControlSpec(
                "edit",
                tag="limcontrib",
                value=f"{float(dataset.get('xmin', 0)) * 1000:g} {float(dataset.get('xmax', 0)) * 1000:g}",
            ),
            ControlSpec("text", "Number of largest contributing components to plot (7):"),
            ControlSpec("edit", tag="compsplot", value="7"),
            ControlSpec("text", "Else plot these component numbers only (Ex: 2:4,7):"),
            ControlSpec("edit", tag="components", value=""),
            ControlSpec("text", "Component numbers to remove from data before plotting:"),
            ControlSpec("edit", tag="subcomps", value=""),
            ControlSpec("text", "Plot title:"),
            ControlSpec("edit", tag="title", value=_envtopo_title(dataset)),
            ControlSpec("text", "Optional topoplot() and envtopo() arguments:"),
            ControlSpec("edit", tag="options", value="'electrodes', 'off'"),
        ]
    )
    return DialogSpec(
        title="Plot component and ERP envelopes -- pop_envtopo()",
        controls=tuple(controls),
        geometry=tuple((2, 1) for _ in range(len(controls) // 2)),
        function_name="pop_envtopo",
        eeglab_source="functions/popfunc/pop_envtopo.m",
        help_text="pophelp('pop_envtopo')",
        size=(825, 369 if not compare else 420),
    )


def _run_gui(
    EEG: dict[str, Any] | list[dict[str, Any]], *, renderer: Any | None = None, compare: bool = False
) -> dict[str, Any] | None:
    result = inputgui(pop_envtopo_dialog_spec(EEG, compare=compare), renderer=renderer)
    if result is None:
        return None
    parsed: dict[str, Any] = {
        "timerange": numeric_vector(result.get("timerange", [])).tolist(),
        "options": {
            "components": numeric_vector(result.get("components", []), dtype=int).tolist(),
            "limcontrib": numeric_vector(result.get("limcontrib", [])).tolist(),
            "compsplot": numeric_vector(result.get("compsplot", []), dtype=int).tolist(),
            # Blank field -> 0 ("remove none"), matching EEGLAB; [] would mean "remove all but compnums".
            "subcomps": numeric_vector(result.get("subcomps", []), dtype=int).tolist() or 0,
            "title": str(result.get("title", "") or ""),
            "options": str(result.get("options", "") or ""),
        },
    }
    if compare:
        count = len(EEG) if isinstance(EEG, list) else 1
        parsed["subindices"] = _parse_subindices(result.get("subindices", ""), count)
    return parsed


def _envtopo_title(EEG: dict[str, Any]) -> str:
    name = str(EEG.get("setname") or "").strip()
    if name:
        return f"Largest ERP components of {name}"
    return "Largest ERP components"


def _parse_subindices(value: Any, count: int) -> tuple[int, int]:
    indices = numeric_vector(value, dtype=int)
    if indices.size < 2:
        raise ValueError("pop_envtopo: enter two dataset indices to subtract, for example 2 1")
    if np.any(indices < 1) or np.any(indices > count):
        raise ValueError(f"pop_envtopo: dataset indices must be within 1..{count}")
    return int(indices[0]), int(indices[1])


def _component_mean(EEG: dict[str, Any], timerange: Any) -> dict[str, Any]:
    data, times = data_time_slice(EEG, timerange)
    icaweights = np.asarray(EEG.get("icaweights", []), dtype=float)
    icasphere = np.asarray(EEG.get("icasphere", []), dtype=float)
    if icaweights.size == 0 or icasphere.size == 0:
        raise ValueError("pop_envtopo requires ICA weights")
    icachansind = component_channel_indices(EEG, data.shape[0])
    data = data[icachansind, :, :]
    weights = finite_matmul(icaweights, icasphere)
    if weights.shape[1] != data.shape[0]:
        raise ValueError("ICA weights do not match EEG.icachansind channel count")
    maps = component_maps(EEG)
    return {
        "mean": np.nanmean(data, axis=2),
        "times": times,
        "weights": weights,
        "maps": maps,
        "chanlocs": _component_chanlocs(EEG, maps, icachansind),
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

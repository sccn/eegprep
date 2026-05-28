"""EEGLAB-style ERP comparison plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._plot_utils import (
    as_eeg_list,
    component_activations,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
)


def pop_comperp(
    ALLEEG: Any = None,
    flag: int = 1,
    datadd: Any = None,
    datsub: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Compute and plot grand-average ERPs across loaded datasets."""
    datasets = as_eeg_list(ALLEEG)
    if gui is None:
        gui = datadd is None
    if gui:
        result = _run_gui(datasets, flag=int(flag), renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        datadd = result["datadd"]
        datsub = result["datsub"]
        kwargs.update(result["options"])
    add_indices = _dataset_indices(datadd, len(datasets))
    sub_indices = _dataset_indices(datsub, len(datasets), allow_empty=True)
    erp1 = _grand_erp([datasets[index] for index in add_indices], int(flag), kwargs.get("chans"))
    erp2 = (
        _grand_erp([datasets[index] for index in sub_indices], int(flag), kwargs.get("chans"))
        if sub_indices.size
        else None
    )
    erpsub = erp1 - erp2 if erp2 is not None else None
    times = eeg_times_ms(datasets[int(add_indices[0])])
    figure = _plot_comperp(erp1, erp2, erpsub, times, title=str(kwargs.get("title") or "ERP grand average"))
    result = {"erp1": erp1, "erp2": erp2, "erpsub": erpsub, "times": times, "figure": figure}
    command = history_command(
        "pop_comperp", int(flag), (add_indices + 1).tolist(), (sub_indices + 1).tolist(), eeg_name="ALLEEG", **kwargs
    )
    return (result, command) if return_com else result


def pop_comperp_dialog_spec(_datasets: list[dict[str, Any]], *, flag: int = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_comperp``."""
    label = "Channels" if int(flag) else "Components"
    return DialogSpec(
        title="ERP grand average/RMS - pop_comperp()",
        controls=(
            ControlSpec("spacer"),
            ControlSpec("text", "avg.        std.      all ERPs"),
            ControlSpec("text", "Datasets to average (ex: 1 3 4):"),
            ControlSpec("edit", tag="datadd", value=""),
            ControlSpec("checkbox", tag="addavg", value=True),
            ControlSpec("checkbox", tag="addstd", value=False),
            ControlSpec("checkbox", tag="addall", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Datasets to average and subtract (ex: 5 6 7):"),
            ControlSpec("edit", tag="datsub", value=""),
            ControlSpec("checkbox", tag="subavg", value=True),
            ControlSpec("checkbox", tag="substd", value=False),
            ControlSpec("checkbox", tag="suball", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Plot difference"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="diffavg", value=True),
            ControlSpec("checkbox", tag="diffstd", value=False),
            ControlSpec("checkbox", tag="diffall", value=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", f"{label} subset ([] = all):"),
            ControlSpec("edit", tag="chans", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Highlight significant regions (.01 -> p=.01)"),
            ControlSpec("edit", tag="alpha", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Use RMS instead of average (check):"),
            ControlSpec("checkbox", tag="mode_rms", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Low pass (Hz) (for display only)"),
            ControlSpec("edit", tag="lowpass", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Plottopo options ('key', 'val'):"),
            ControlSpec("edit", tag="tplotopt", value="'ydir', -1"),
            ControlSpec("pushbutton", "Help"),
        ),
        geometry=(
            (2.6, 0.95),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1,),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 0.25, 1.75),
        ),
        function_name="pop_comperp",
        eeglab_source="functions/popfunc/pop_comperp.m",
        help_text="pophelp('pop_comperp')",
        size=(938, 476),
    )


def _run_gui(datasets: list[dict[str, Any]], *, flag: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_comperp_dialog_spec(datasets, flag=flag), renderer=renderer)
    if result is None:
        return None
    return {
        "datadd": numeric_vector(result.get("datadd", []), dtype=int).tolist(),
        "datsub": numeric_vector(result.get("datsub", []), dtype=int).tolist(),
        "options": {
            "chans": numeric_vector(result.get("chans", []), dtype=int).tolist(),
            "alpha": numeric_vector(result.get("alpha", [])).tolist(),
            "mode": "rms" if bool(result.get("mode_rms", False)) else "ave",
            "lowpass": numeric_vector(result.get("lowpass", [])).tolist(),
            "addavg": "on" if bool(result.get("addavg", True)) else "off",
            "addstd": "on" if bool(result.get("addstd", False)) else "off",
            "addall": "on" if bool(result.get("addall", False)) else "off",
            "subavg": "on" if bool(result.get("subavg", True)) else "off",
            "substd": "on" if bool(result.get("substd", False)) else "off",
            "suball": "on" if bool(result.get("suball", False)) else "off",
            "diffavg": "on" if bool(result.get("diffavg", True)) else "off",
            "diffstd": "on" if bool(result.get("diffstd", False)) else "off",
            "diffall": "on" if bool(result.get("diffall", False)) else "off",
            "tplotopt": str(result.get("tplotopt", "") or ""),
        },
    }


def _dataset_indices(values: Any, count: int, *, allow_empty: bool = False) -> np.ndarray:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        if allow_empty:
            return np.asarray([], dtype=int)
        raise ValueError("Dataset list cannot be empty")
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"Dataset indices must be 1-based and within 1..{count}")
    return vector - 1


def _grand_erp(datasets: list[dict[str, Any]], flag: int, chans: Any) -> np.ndarray:
    erps = []
    for eeg in datasets:
        values = eeg_epoch_data(eeg) if flag else component_activations(eeg)
        if int(eeg.get("trials", 1) or 1) <= 1:
            raise ValueError("pop_comperp requires epoched datasets")
        selected = _selected_rows(chans, values.shape[0])
        erps.append(np.nanmean(values[selected, :, :], axis=2))
    return np.nanmean(np.stack(erps, axis=0), axis=0)


def _selected_rows(values: Any, count: int) -> np.ndarray:
    vector = numeric_vector(values, dtype=int)
    if vector.size == 0:
        return np.arange(count)
    if np.any(vector < 1) or np.any(vector > count):
        raise ValueError(f"Selected channels/components must be within 1..{count}")
    return vector - 1


def _plot_comperp(
    erp1: np.ndarray, erp2: np.ndarray | None, erpsub: np.ndarray | None, times: np.ndarray, *, title: str
):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(times, np.nanmean(erp1, axis=0), color="blue", label="add")
    if erp2 is not None:
        ax.plot(times, np.nanmean(erp2, axis=0), color="red", label="subtract")
    if erpsub is not None:
        ax.plot(times, np.nanmean(erpsub, axis=0), color="black", label="difference")
    ax.axhline(0, color="0.7", linewidth=0.6)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("uV")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


__all__ = ["pop_comperp", "pop_comperp_dialog_spec"]

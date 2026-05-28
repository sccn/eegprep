"""EEGLAB-style ERP comparison plotting."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

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
    _raise_for_unsupported_options(kwargs)
    if sub_indices.size and sub_indices.size != add_indices.size:
        raise ValueError("datadd and datsub must contain the same number of datasets")
    selected_datasets = [datasets[index] for index in np.union1d(add_indices, sub_indices)]
    _validate_time_grid(selected_datasets)
    mode = str(kwargs.get("mode") or "ave").lower()
    erp1 = _grand_erp([datasets[index] for index in add_indices], int(flag), kwargs.get("chans"), mode)
    erp2 = (
        _grand_erp([datasets[index] for index in sub_indices], int(flag), kwargs.get("chans"), mode)
        if sub_indices.size
        else None
    )
    erpsub = erp1 - erp2 if erp2 is not None else None
    lowpass = numeric_vector(kwargs.get("lowpass", []))
    if lowpass.size:
        erp1 = _lowpass_erp(erp1, float(lowpass[0]), float(datasets[int(add_indices[0])].get("srate", 1) or 1))
        if erp2 is not None:
            erp2 = _lowpass_erp(erp2, float(lowpass[0]), float(datasets[int(add_indices[0])].get("srate", 1) or 1))
        if erpsub is not None:
            erpsub = _lowpass_erp(erpsub, float(lowpass[0]), float(datasets[int(add_indices[0])].get("srate", 1) or 1))
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
    _raise_for_unsupported_gui_options(result)
    return {
        "datadd": numeric_vector(result.get("datadd", []), dtype=int).tolist(),
        "datsub": numeric_vector(result.get("datsub", []), dtype=int).tolist(),
        "options": {
            "chans": numeric_vector(result.get("chans", []), dtype=int).tolist(),
            "mode": "rms" if bool(result.get("mode_rms", False)) else "ave",
            "lowpass": numeric_vector(result.get("lowpass", [])).tolist(),
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


def _grand_erp(datasets: list[dict[str, Any]], flag: int, chans: Any, mode: str) -> np.ndarray:
    erps = []
    for eeg in datasets:
        values = eeg_epoch_data(eeg) if flag else component_activations(eeg)
        if int(eeg.get("trials", 1) or 1) <= 1:
            raise ValueError("pop_comperp requires epoched datasets")
        selected = _selected_rows(chans, values.shape[0])
        erps.append(np.nanmean(values[selected, :, :], axis=2))
    stacked = np.stack(erps, axis=0)
    if mode == "rms":
        return np.sqrt(np.nanmean(stacked * stacked, axis=0))
    if mode != "ave":
        raise ValueError("mode must be 'ave' or 'rms'")
    return np.nanmean(stacked, axis=0)


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


def _validate_time_grid(datasets: list[dict[str, Any]]) -> None:
    if not datasets:
        return
    reference = (
        int(datasets[0].get("pnts", 0) or 0),
        float(datasets[0].get("srate", 0) or 0),
        float(datasets[0].get("xmin", 0) or 0),
        float(datasets[0].get("xmax", 0) or 0),
    )
    for index, eeg in enumerate(datasets[1:], start=2):
        candidate = (
            int(eeg.get("pnts", 0) or 0),
            float(eeg.get("srate", 0) or 0),
            float(eeg.get("xmin", 0) or 0),
            float(eeg.get("xmax", 0) or 0),
        )
        if candidate != reference:
            raise ValueError(f"Dataset {index} does not share the same time grid")


def _lowpass_erp(values: np.ndarray, cutoff: float, srate: float) -> np.ndarray:
    if cutoff <= 0 or cutoff >= srate / 2:
        raise ValueError("lowpass must be greater than 0 and below Nyquist")
    sos = butter(4, cutoff, btype="lowpass", fs=srate, output="sos")
    return sosfiltfilt(sos, values, axis=1)


def _raise_for_unsupported_gui_options(result: dict[str, Any]) -> None:
    if numeric_vector(result.get("alpha", [])).size:
        raise NotImplementedError("pop_comperp does not yet support significance highlighting")
    unsupported_on = {
        "addstd": "add standard deviation",
        "addall": "plot all added ERPs",
        "substd": "subtract standard deviation",
        "suball": "plot all subtracted ERPs",
        "diffstd": "difference standard deviation",
        "diffall": "plot all differences",
    }
    for field, label in unsupported_on.items():
        if bool(result.get(field, False)):
            raise NotImplementedError(f"pop_comperp does not yet support {label}")
    if not bool(result.get("addavg", True)):
        raise NotImplementedError("pop_comperp currently requires plotting the added average")
    if not bool(result.get("subavg", True)):
        raise NotImplementedError("pop_comperp currently requires plotting the subtracted average")
    if not bool(result.get("diffavg", True)):
        raise NotImplementedError("pop_comperp currently requires plotting the difference average")


def _raise_for_unsupported_options(options: dict[str, Any]) -> None:
    supported = {"chans", "mode", "lowpass", "title"}
    unsupported = sorted(key for key in set(options) - supported if not _is_default_off(options[key]))
    if unsupported:
        raise NotImplementedError(f"pop_comperp does not yet support option(s): {', '.join(unsupported)}")


def _is_default_off(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.lower() == "off" or value == ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(np.asarray(value).ravel()) == 0
    return value is False


__all__ = ["pop_comperp", "pop_comperp_dialog_spec"]

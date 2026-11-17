"""EEGLAB-style ERP comparison plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import ttest_1samp, ttest_rel
from scipy.signal import butter, sosfiltfilt

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import (
    as_eeg_list,
    component_activations,
    eeg_epoch_data,
    eeg_times_ms,
    history_command,
    numeric_vector,
    parse_plot_options_text,
    show_figures,
)
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import is_on
from eegprep.functions.sigprocfunc.plottopo import plottopo


def pop_comperp(
    ALLEEG: Any = None,
    flag: int = 1,
    datadd: Any = None,
    datsub: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
    **kwargs: Any,
):
    """Compute and plot grand-average ERPs across loaded datasets.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
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
    if sub_indices.size and sub_indices.size != add_indices.size:
        raise ValueError("datadd and datsub must contain the same number of datasets")
    options = _normalise_options(kwargs, has_sub=bool(sub_indices.size))
    selected_datasets = [datasets[index] for index in np.union1d(add_indices, sub_indices)]
    _validate_time_grid(selected_datasets)
    mode = str(options["mode"]).lower()
    add_stack = _dataset_erps([datasets[index] for index in add_indices], int(flag), options.get("chans"))
    sub_stack = (
        _dataset_erps([datasets[index] for index in sub_indices], int(flag), options.get("chans"))
        if sub_indices.size
        else None
    )
    diff_stack = add_stack - sub_stack if sub_stack is not None else None
    erp1 = _collapse_erps(add_stack, mode)
    erp2 = _collapse_erps(sub_stack, mode) if sub_stack is not None else None
    erpsub = erp1 - erp2 if erp2 is not None else None
    pvalues = _significance(add_stack, sub_stack, options.get("alpha"))
    lowpass = numeric_vector(options.get("lowpass", []))
    if lowpass.size:
        cutoff = float(lowpass[0])
        srate = float(datasets[int(add_indices[0])].get("srate", 1) or 1)
        erp1 = _lowpass(erp1, cutoff, srate, axis=1)
        if erp2 is not None:
            erp2 = _lowpass(erp2, cutoff, srate, axis=1)
        if erpsub is not None:
            erpsub = _lowpass(erpsub, cutoff, srate, axis=1)
        add_stack = _lowpass(add_stack, cutoff, srate, axis=2)
        if sub_stack is not None:
            sub_stack = _lowpass(sub_stack, cutoff, srate, axis=2)
        if diff_stack is not None:
            diff_stack = _lowpass(diff_stack, cutoff, srate, axis=2)
    times = eeg_times_ms(datasets[int(add_indices[0])])
    _apply_single_dataset_rules(options, add_stack.shape[0], 0 if sub_stack is None else sub_stack.shape[0])
    figure = _plot_comperp(
        erp1,
        erp2,
        erpsub,
        add_stack,
        sub_stack,
        diff_stack,
        times,
        pvalues=pvalues,
        options=options,
        chanlocs=_plot_chanlocs(datasets[int(add_indices[0])], int(flag), options.get("chans")),
        rect=int(flag) == 0 or str(options.get("geom") or "").lower() == "array",
    )
    result = {"erp1": erp1, "erp2": erp2, "erpsub": erpsub, "times": times, "pvalues": pvalues, "figure": figure}
    show_figures(figure, plot=plot)
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
            ControlSpec("spacer"),
            ControlSpec("text", "avg."),
            ControlSpec("text", "std."),
            ControlSpec("text", "all ERPs"),
            ControlSpec("spacer"),
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
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1.1, 0.8, 0.21, 0.21, 0.21, 0.1),
            (1,),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
            (1.48, 1.03, 1),
        ),
        function_name="pop_comperp",
        eeglab_source="functions/popfunc/pop_comperp.m",
        help_text="pophelp('pop_comperp')",
        size=(938, 476),
        row_spacing=8,
    )


def _run_gui(datasets: list[dict[str, Any]], *, flag: int, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_comperp_dialog_spec(datasets, flag=flag), renderer=renderer)
    if result is None:
        return None
    alpha = numeric_vector(result.get("alpha", []))
    return {
        "datadd": numeric_vector(result.get("datadd", []), dtype=int).tolist(),
        "datsub": numeric_vector(result.get("datsub", []), dtype=int).tolist(),
        "options": {
            "chans": numeric_vector(result.get("chans", []), dtype=int).tolist(),
            "mode": "rms" if bool(result.get("mode_rms", False)) else "ave",
            "lowpass": numeric_vector(result.get("lowpass", [])).tolist(),
            "alpha": float(alpha[0]) if alpha.size else [],
            "addavg": "on" if bool(result.get("addavg", True)) else "off",
            "addstd": "on" if bool(result.get("addstd", False)) else "off",
            "addall": "on" if bool(result.get("addall", False)) else "off",
            "subavg": "on" if bool(result.get("subavg", True)) else "off",
            "substd": "on" if bool(result.get("substd", False)) else "off",
            "suball": "on" if bool(result.get("suball", False)) else "off",
            "diffavg": "on" if bool(result.get("diffavg", True)) else "off",
            "diffstd": "on" if bool(result.get("diffstd", False)) else "off",
            "diffall": "on" if bool(result.get("diffall", False)) else "off",
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


def _dataset_erps(datasets: list[dict[str, Any]], flag: int, chans: Any) -> np.ndarray:
    erps = []
    for eeg in datasets:
        values = eeg_epoch_data(eeg) if flag else component_activations(eeg)
        if int(eeg.get("trials", 1) or 1) <= 1:
            raise ValueError("pop_comperp requires epoched datasets")
        selected = _selected_rows(chans, values.shape[0])
        erps.append(np.nanmean(values[selected, :, :], axis=2))
    return np.stack(erps, axis=0)


def _collapse_erps(stacked: np.ndarray, mode: str) -> np.ndarray:
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


def _apply_single_dataset_rules(options: dict[str, Any], add_count: int, sub_count: int) -> None:
    if add_count == 1 and is_on(options.get("addavg")):
        options["addall"] = "on"
        options["addavg"] = "off"
    if sub_count == 1 and is_on(options.get("subavg")):
        options["suball"] = "on"
        options["subavg"] = "off"
    if add_count == 1 and sub_count == 1 and is_on(options.get("diffavg")):
        options["diffall"] = "on"
        options["diffavg"] = "off"


def _plot_chanlocs(dataset: dict[str, Any], flag: int, chans: Any) -> list[dict[str, Any]]:
    if not flag:
        count = int(np.asarray(dataset.get("icaweights", [])).shape[0] or dataset.get("nbchan", 0) or 0)
        selected = _selected_rows(chans, count) if count else np.asarray([], dtype=int)
        return [{"labels": str(int(index) + 1)} for index in selected]
    locs = chanlocs_as_list(dataset.get("chanlocs", []))
    if not locs:
        return []
    selected = _selected_rows(chans, len(locs))
    return [locs[int(index)] for index in selected]


def _plot_comperp(
    erp1: np.ndarray,
    erp2: np.ndarray | None,
    erpsub: np.ndarray | None,
    add_stack: np.ndarray,
    sub_stack: np.ndarray | None,
    diff_stack: np.ndarray | None,
    times: np.ndarray,
    *,
    pvalues: np.ndarray | None,
    options: dict[str, Any],
    chanlocs: list[dict[str, Any]],
    rect: bool,
):
    plot_times, mask = _plot_time_mask(times, options.get("tlim"))
    mode = str(options.get("mode") or "ave").lower()
    traces: list[np.ndarray] = []
    styles: list[dict[str, Any]] = []

    def add_trace(values: np.ndarray, style: dict[str, Any]) -> None:
        traces.append(np.asarray(values, dtype=float)[:, mask])
        styles.append(style)

    def add_stack_traces(stack: np.ndarray, color: str, label: str) -> None:
        for index, erp in enumerate(stack):
            add_trace(erp, {"color": color, "linewidth": 0.8, "label": label if index == 0 else None})

    def add_std(stack: np.ndarray, center: np.ndarray, color: str, label: str) -> None:
        spread = np.nanstd(stack, axis=0, ddof=1 if stack.shape[0] > 1 else 0)
        if mode == "rms":
            center = _collapse_erps(stack, mode)
        add_trace(center + spread, {"color": color, "linestyle": ":", "linewidth": 1.0, "label": label})
        add_trace(center - spread, {"color": color, "linestyle": ":", "linewidth": 1.0, "label": None})

    if is_on(options.get("addall")):
        add_stack_traces(add_stack, "b", "Add")
    if is_on(options.get("suball")) and sub_stack is not None:
        add_stack_traces(sub_stack, "r", "Subtract")
    if is_on(options.get("diffall")) and diff_stack is not None:
        add_stack_traces(diff_stack, "k", "Difference")
    average_label = "RMS" if mode == "rms" else "Average"
    if is_on(options.get("addavg")):
        add_trace(erp1, {"color": "b", "linewidth": 2.0, "label": f"Add {average_label}"})
    if erp2 is not None and is_on(options.get("subavg")):
        add_trace(erp2, {"color": "r", "linewidth": 2.0, "label": f"Subtract {average_label}"})
    if erpsub is not None and is_on(options.get("diffavg")):
        add_trace(erpsub, {"color": "k", "linewidth": 2.0, "label": f"Difference {average_label}"})
    if is_on(options.get("addstd")):
        add_std(add_stack, erp1, "b", "Add std")
    if is_on(options.get("substd")) and sub_stack is not None and erp2 is not None:
        add_std(sub_stack, erp2, "r", "Subtract std")
    if is_on(options.get("diffstd")) and diff_stack is not None and erpsub is not None:
        add_std(diff_stack, erpsub, "k", "Difference std")
    if not traces:
        add_trace(erp1, {"color": "b", "linewidth": 1.2, "label": "Add"})
    data = np.stack(traces, axis=2)
    regions = None
    alpha = options.get("alpha")
    if pvalues is not None and alpha is not None:
        regions = np.asarray(pvalues, dtype=float)[:, mask] < float(alpha)
    ydir = int(parse_plot_options_text(str(options.get("tplotopt") or "")).get("ydir", -1))
    return plottopo(
        data,
        times=plot_times,
        chanlocs=chanlocs,
        title=str(options.get("title") or "ERP grand average"),
        ydir=ydir,
        ylimits=options.get("ylim"),
        rect=rect or not chanlocs,
        overlay=True,
        styles=styles,
        regions=regions,
    )


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


def _lowpass(values: np.ndarray, cutoff: float, srate: float, axis: int) -> np.ndarray:
    if cutoff <= 0 or cutoff >= srate / 2:
        raise ValueError("lowpass must be greater than 0 and below Nyquist")
    sos = butter(4, cutoff, btype="lowpass", fs=srate, output="sos")
    return sosfiltfilt(sos, values, axis=axis)


def _is_default_off(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "off", "no", "false", "0"}
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value).ravel()
        return len(array) == 0 or (len(array) == 1 and not bool(array[0]))
    return not bool(value)


def _normalise_options(options: dict[str, Any], *, has_sub: bool) -> dict[str, Any]:
    supported = {
        "chans",
        "mode",
        "lowpass",
        "title",
        "alpha",
        "geom",
        "tlim",
        "ylim",
        "tplotopt",
        "addstd",
        "substd",
        "diffstd",
        "addavg",
        "subavg",
        "diffavg",
        "addall",
        "suball",
        "diffall",
        "std",
        "diffonly",
        "allerps",
        "multcmp",
    }
    unknown = sorted(key for key in set(options) - supported if not _is_default_off(options[key]))
    if unknown:
        raise ValueError(f"pop_comperp unsupported option(s): {', '.join(unknown)}")
    normalised = {
        "chans": options.get("chans"),
        "mode": str(options.get("mode") or "ave").lower(),
        "lowpass": options.get("lowpass", []),
        "title": options.get("title") or "ERP grand average",
        "alpha": _optional_alpha(options.get("alpha")),
        "tlim": options.get("tlim", []),
        "ylim": options.get("ylim", []),
        "geom": str(options.get("geom") or ""),
        "tplotopt": str(options.get("tplotopt") or ""),
        "addavg": _onoff_option(options.get("addavg"), True),
        "subavg": _onoff_option(options.get("subavg"), has_sub),
        "diffavg": _onoff_option(options.get("diffavg"), has_sub),
        "addstd": _onoff_option(options.get("addstd"), False),
        "substd": _onoff_option(options.get("substd"), False),
        "diffstd": _onoff_option(options.get("diffstd"), False),
        "addall": _onoff_option(options.get("addall"), False),
        "suball": _onoff_option(options.get("suball"), False),
        "diffall": _onoff_option(options.get("diffall"), False),
    }
    std = str(options.get("std", "none")).lower()
    if std != "none":
        if has_sub:
            normalised["diffstd"] = std
        else:
            normalised["addstd"] = std
    allerps = str(options.get("allerps", "none")).lower()
    if allerps != "none":
        normalised["diffall" if has_sub else "addall"] = allerps
    diffonly = str(options.get("diffonly", "none")).lower()
    if diffonly == "on" and has_sub:
        normalised.update({"addavg": "off", "subavg": "off", "addall": "off", "suball": "off", "diffavg": "on"})
    elif diffonly == "off" and has_sub:
        normalised.update({"addavg": "on", "subavg": "on"})
    return normalised


def _optional_alpha(value: Any) -> float | None:
    values = numeric_vector(value if value is not None else [])
    if values.size == 0:
        return None
    alpha = float(values[0])
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    return alpha


def _onoff_option(value: Any, default: bool) -> str:
    if value is None or (_is_default_off(value) and default is False):
        return "on" if default else "off"
    return "on" if is_on(value) else "off"


def _significance(add_stack: np.ndarray, sub_stack: np.ndarray | None, alpha: float | None) -> np.ndarray | None:
    if alpha is None:
        return None
    if add_stack.shape[0] <= 1:
        raise ValueError("T-tests require more than one datadd dataset")
    if sub_stack is not None:
        if sub_stack.shape[0] <= 1:
            raise ValueError("Paired T-tests require more than one datsub dataset")
        return np.asarray(ttest_rel(add_stack, sub_stack, axis=0, nan_policy="omit").pvalue)
    return np.asarray(ttest_1samp(add_stack, popmean=0.0, axis=0, nan_policy="omit").pvalue)


def _plot_time_mask(times: np.ndarray, tlim: Any) -> tuple[np.ndarray, np.ndarray]:
    values = numeric_vector(tlim if tlim is not None else [])
    if values.size != 2:
        return times, np.ones(times.shape, dtype=bool)
    mask = (times >= values[0]) & (times <= values[1])
    if not np.any(mask):
        raise ValueError("tlim does not include any samples")
    return times[mask], mask


__all__ = ["pop_comperp", "pop_comperp_dialog_spec"]

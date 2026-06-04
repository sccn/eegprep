"""Reject epochs with out-of-bounds data or ICA activation values."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import open_epoched_rejection_browser
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import (
    copy_eeg,
    eegthresh_marks,
    one_based_indices,
    parse_numeric_sequence,
    rejection_data,
    update_reject_fields,
)
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch


def pop_eegthresh(
    EEG: dict[str, Any] | list[dict[str, Any]],
    icacomp: int | bool = 1,
    elecrange: Any = None,
    negthresh: Any = None,
    posthresh: Any = None,
    starttime: Any = None,
    endtime: Any = None,
    superpose: int | bool = 0,
    reject: int | bool = 1,
    topcommand: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
):
    """Mark or reject epochs using EEGLAB ``pop_eegthresh`` semantics."""
    if EEG is None:
        return (None, "") if return_com else (None, [])
    if gui is None:
        gui = elecrange is None and negthresh is None and posthresh is None
    if isinstance(EEG, list):
        outputs = []
        rejected = []
        if gui:
            result = _run_gui(EEG[0], int(bool(icacomp)), renderer=renderer)
            if result is None:
                return (EEG, "") if return_com else (EEG, [])
            elecrange, negthresh, posthresh, starttime, endtime, superpose, reject = result
        for dataset in EEG:
            out, inds, _command = _apply_one(
                dataset,
                icacomp,
                elecrange,
                negthresh,
                posthresh,
                starttime,
                endtime,
                superpose,
                reject,
                display=False,
            )
            outputs.append(out)
            rejected.append(inds)
        command = _history_command(icacomp, elecrange, negthresh, posthresh, starttime, endtime, superpose, reject)
        return (outputs, command) if return_com else (outputs, rejected)
    if gui:
        result = _run_gui(EEG, int(bool(icacomp)), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else (EEG, [])
        elecrange, negthresh, posthresh, starttime, endtime, superpose, reject = result
    out, rejected, command = _apply_one(
        EEG,
        icacomp,
        elecrange,
        negthresh,
        posthresh,
        starttime,
        endtime,
        superpose,
        reject,
        display=bool(gui or topcommand is not None),
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else (out, rejected)


def pop_eegthresh_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_eegthresh``."""
    is_data = int(bool(icacomp))
    rows = int(EEG.get("nbchan", 0) or 0) if is_data else int(np.asarray(EEG.get("icaweights", [])).shape[0])
    title = (
        "Rejection abnormal elec. values -- pop_eegthresh()"
        if is_data
        else "Rejection abnormal comp. values -- pop_eegthresh()"
    )
    label = "Electrode (indices(s), Ex: 2 4 5):" if is_data else "Component (indices, Ex: 2 6:8 10):"
    unit = "uV" if is_data else "std. dev"
    return DialogSpec(
        title=title,
        function_name="pop_eegthresh",
        eeglab_source="functions/popfunc/pop_eegthresh.m",
        help_text="pophelp('pop_eegthresh')",
        size=(600, 318),
        geometry=((1, 0.1, 0.75),) * 5 + (1,) + ((1, 0.22, 0.85),) * 2,
        controls=(
            ControlSpec("text", label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="elecrange", value=f"1:{rows}"),
            ControlSpec("text", f"Minimum rejection threshold(s) ({unit}, Ex:-20 -10 -15):"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="negthresh", value="-10" if is_data else "-20"),
            ControlSpec("text", f"Maximum rejection threshold(s) ({unit}, Ex: 20 10 15):"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="posthresh", value="10" if is_data else "20"),
            ControlSpec("text", "Start time limit(s) (seconds, Ex -0.1 0.3):"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="starttime", value=f"{float(EEG.get('xmin', 0)):g}"),
            ControlSpec("text", "End time limit(s) (seconds, Ex 0.2):"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="endtime", value=f"{float(EEG.get('xmax', 0)):g}"),
            ControlSpec("spacer"),
            ControlSpec("text", "Display previous rejection marks"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="superpose", value=False),
            ControlSpec("text", "Reject marked trial(s)"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="reject", value=False),
        ),
    )


def _run_gui(EEG: dict[str, Any], icacomp: int, renderer: Any | None) -> tuple[Any, ...] | None:
    result = inputgui(pop_eegthresh_dialog_spec(EEG, icacomp), renderer=renderer)
    if result is None:
        return None
    return (
        result.get("elecrange", ""),
        result.get("negthresh", ""),
        result.get("posthresh", ""),
        result.get("starttime", ""),
        result.get("endtime", ""),
        int(bool(result.get("superpose", False))),
        int(bool(result.get("reject", False))),
    )


def _apply_one(
    EEG: dict[str, Any],
    icacomp: int | bool,
    elecrange: Any,
    negthresh: Any,
    posthresh: Any,
    starttime: Any,
    endtime: Any,
    superpose: int | bool,
    reject: int | bool,
    *,
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], list[int], str]:
    out = copy_eeg(EEG)
    data, row_count = rejection_data(out, icacomp)
    trials = int(out.get("trials", data.shape[2]) or data.shape[2])
    if trials <= 1:
        raise ValueError("pop_eegthresh requires epoched data")
    elecrange = one_based_indices(elecrange, limit=row_count, default_all=True)
    negthresh = [-10.0] if negthresh is None else negthresh
    posthresh = [10.0] if posthresh is None else posthresh
    starttime = [float(out.get("xmin", 0.0))] if starttime is None else starttime
    endtime = [float(out.get("xmax", 0.0))] if endtime is None else endtime
    marks, marks_e = eegthresh_marks(
        data,
        elecrange,
        negthresh,
        posthresh,
        (float(out.get("xmin", 0.0)), float(out.get("xmax", 0.0))),
        starttime,
        endtime,
    )
    update_reject_fields(out, icacomp=icacomp, kind="rejthresh", reject=marks, reject_e=marks_e)
    rejected = (np.flatnonzero(marks) + 1).tolist()
    command = _history_command(
        icacomp, elecrange, negthresh, posthresh, starttime, endtime, superpose, int(bool(reject))
    )
    if display:
        open_epoched_rejection_browser(
            out,
            data=data,
            icacomp=icacomp,
            elecrange=elecrange,
            kind="rejthresh",
            superpose=superpose,
            reject=reject,
            command=command,
            command_callback=command_callback,
            show=show,
        )
    elif int(bool(reject)) and rejected:
        out = pop_rejepoch(out, rejected, 0)
    return out, rejected, command


def _history_command(
    icacomp: int | bool,
    elecrange: Any,
    negthresh: Any,
    posthresh: Any,
    starttime: Any,
    endtime: Any,
    superpose: int | bool,
    reject: int | bool,
) -> str:
    args = [
        int(bool(icacomp)),
        parse_numeric_sequence(elecrange, dtype=int),
        parse_numeric_sequence(negthresh, dtype=float),
        parse_numeric_sequence(posthresh, dtype=float),
        parse_numeric_sequence(starttime, dtype=float),
        parse_numeric_sequence(endtime, dtype=float),
        int(superpose),
        int(bool(reject)),
    ]
    return (
        "EEG = pop_eegthresh(EEG, "
        + ", ".join(format_history_value(arg, cell_for_sequence=None) for arg in args)
        + ");"
    )

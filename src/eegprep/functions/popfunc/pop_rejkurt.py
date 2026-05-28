"""Reject epochs using EEGLAB-style kurtosis statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import (
    copy_eeg,
    kurtosis_marks,
    one_based_indices,
    parse_numeric_sequence,
    rejection_data,
    update_reject_fields,
)
from eegprep.functions.popfunc.pop_rejepoch import pop_rejepoch


def pop_rejkurt(
    EEG: dict[str, Any] | list[dict[str, Any]],
    icacomp: int | bool = 1,
    elecrange: Any = None,
    locthresh: Any = 3,
    globthresh: Any = 3,
    superpose: int | bool = 0,
    reject: int | bool = 1,
    vistype: int = 0,
    topcommand: Any = None,
    plotflag: int | bool = 0,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Mark or reject epochs by local/global kurtosis."""
    if EEG is None:
        return (None, "") if return_com else (None, [], [], 0)
    if gui is None:
        gui = elecrange is None
    if isinstance(EEG, list):
        if gui:
            result = _run_gui(EEG[0], int(bool(icacomp)), renderer=renderer)
            if result is None:
                return (EEG, "") if return_com else (EEG, [], [], 0)
            elecrange, locthresh, globthresh, superpose, reject, vistype = result
        outputs = [
            _apply_one(dataset, icacomp, elecrange, locthresh, globthresh, superpose, reject, vistype)[0]
            for dataset in EEG
        ]
        command = _history_command(icacomp, elecrange, locthresh, globthresh, superpose, reject, vistype, plotflag)
        return (outputs, command) if return_com else (outputs, locthresh, globthresh, 0)
    if gui:
        result = _run_gui(EEG, int(bool(icacomp)), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else (EEG, [], [], 0)
        elecrange, locthresh, globthresh, superpose, reject, vistype = result
    out, local_threshold, global_threshold, rejected, command = _apply_one(
        EEG, icacomp, elecrange, locthresh, globthresh, superpose, reject, vistype
    )
    return (out, command) if return_com else (out, local_threshold, global_threshold, len(rejected))


def pop_rejkurt_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejkurt``."""
    is_data = int(bool(icacomp))
    rows = int(EEG.get("nbchan", 0) or 0) if is_data else int(np.asarray(EEG.get("icaweights", [])).shape[0])
    title = "Data kurtosis rejection -- pop_rejkurt()" if is_data else "Component kurtosis rejection -- pop_rejkurt()"
    row_label = "Electrode (indices; Ex: 2 6:8 10):" if is_data else "Component (indices; Ex: 2 6:8 10):"
    return DialogSpec(
        title=title,
        function_name="pop_rejkurt",
        eeglab_source="functions/popfunc/pop_rejkurt.m",
        help_text="pophelp('pop_rejkurt')",
        size=(600, 284),
        geometry=((1, 0.1, 0.75),) * 3 + (1,) + ((1, 0.22, 0.85),) * 2,
        controls=(
            ControlSpec("text", row_label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="elecrange", value=f"1:{rows}"),
            ControlSpec("text", "Single-channel/component limit(s) (std. dev.)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="locthresh", value="3"),
            ControlSpec("text", "All-channel/component limit(s) (std. dev.)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="globthresh", value="3"),
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
    result = inputgui(pop_rejkurt_dialog_spec(EEG, icacomp), renderer=renderer)
    if result is None:
        return None
    return (
        result.get("elecrange", ""),
        result.get("locthresh", "3"),
        result.get("globthresh", "3"),
        int(bool(result.get("superpose", False))),
        int(bool(result.get("reject", False))),
        0,
    )


def _apply_one(
    EEG: dict[str, Any],
    icacomp: int | bool,
    elecrange: Any,
    locthresh: Any,
    globthresh: Any,
    superpose: int | bool,
    reject: int | bool,
    vistype: int,
) -> tuple[dict[str, Any], list[float], list[float], list[int], str]:
    out = copy_eeg(EEG)
    data, row_count = rejection_data(out, icacomp)
    if int(out.get("trials", data.shape[2]) or data.shape[2]) <= 1:
        raise ValueError("pop_rejkurt requires epoched data")
    elecrange = one_based_indices(elecrange, limit=row_count, default_all=True)
    marks, marks_e, local_scores, global_scores = kurtosis_marks(data, elecrange, locthresh, globthresh)
    out.setdefault("stats", {})
    if int(bool(icacomp)):
        out["stats"]["kurtE"] = local_scores
        out["stats"]["kurt"] = global_scores
    else:
        out["stats"]["icakurtE"] = local_scores
        out["stats"]["icakurt"] = global_scores
    update_reject_fields(out, icacomp=icacomp, kind="rejkurt", reject=marks, reject_e=marks_e)
    rejected = (np.flatnonzero(marks) + 1).tolist()
    if int(bool(reject)) and rejected:
        out = pop_rejepoch(out, rejected, 0)
    command = _history_command(icacomp, elecrange, locthresh, globthresh, superpose, reject, vistype, 0)
    return (
        out,
        parse_numeric_sequence(locthresh, dtype=float),
        parse_numeric_sequence(globthresh, dtype=float),
        rejected,
        command,
    )


def _history_command(
    icacomp: int | bool,
    elecrange: Any,
    locthresh: Any,
    globthresh: Any,
    superpose: int | bool,
    reject: int | bool,
    vistype: int,
    plotflag: int | bool,
) -> str:
    args = [
        int(bool(icacomp)),
        parse_numeric_sequence(elecrange, dtype=int),
        parse_numeric_sequence(locthresh, dtype=float),
        parse_numeric_sequence(globthresh, dtype=float),
        int(bool(superpose)),
        int(bool(reject)),
        int(vistype),
        [],
        int(bool(plotflag)),
    ]
    return (
        "EEG = pop_rejkurt(EEG, " + ", ".join(format_history_value(arg, cell_for_sequence=None) for arg in args) + ");"
    )

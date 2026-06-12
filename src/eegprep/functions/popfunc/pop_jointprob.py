"""Reject epochs using EEGLAB-style joint probability."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import run_epoched_rejection, vistype_from_gui
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import jointprob_marks, parse_numeric_sequence


def pop_jointprob(
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
    command_callback: Any | None = None,
    show: bool = True,
):
    """Mark or reject epochs by local/global joint probability."""
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
            _apply_one(
                dataset,
                icacomp,
                elecrange,
                locthresh,
                globthresh,
                superpose,
                reject,
                vistype,
                display=False,
            )[0]
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
        EEG,
        icacomp,
        elecrange,
        locthresh,
        globthresh,
        superpose,
        reject,
        vistype,
        display=bool(int(vistype) == 1 or topcommand is not None),
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else (out, local_threshold, global_threshold, len(rejected))


def pop_jointprob_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_jointprob``."""
    is_data = int(bool(icacomp))
    rows = int(EEG.get("nbchan", 0) or 0) if is_data else int(np.asarray(EEG.get("icaweights", [])).shape[0])
    title = "Reject improbable data -- pop_jointprob()" if is_data else "Reject. improbable comp. -- pop_jointprob()"
    row_label = "Electrode (indices; Ex: 2 6:8 10):" if is_data else "Component (indices; Ex: 2 6:8 10):"
    local_label = (
        "Single-channel limit(s) (std. dev(s).: Ex: 2 2 2.5):"
        if is_data
        else "Single-component limit(s) (std. dev(s).: Ex: 2 2 2.5):"
    )
    global_label = (
        "All-channel limit(s) (std. dev(s).: Ex: 2 2.1 2):"
        if is_data
        else "All-component limit(s) (std. dev(s).: Ex: 2 2.1 2):"
    )
    threshold_default = "3" if is_data else "5"
    return DialogSpec(
        title=title,
        function_name="pop_jointprob",
        eeglab_source="functions/popfunc/pop_jointprob.m",
        help_text="pophelp('pop_jointprob')",
        size=(600, 320),
        geometry=((1, 0.1, 0.75),) * 3 + ((1, 0.26, 0.9),) + (1,) + ((1, 0.22, 0.85),) * 2,
        controls=(
            ControlSpec("text", row_label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="elecrange", value=f"1:{rows}"),
            ControlSpec("text", local_label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="locthresh", value=threshold_default),
            ControlSpec("text", global_label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="globthresh", value=threshold_default),
            ControlSpec("text", "Visualization type"),
            ControlSpec("spacer"),
            ControlSpec("popupmenu", "REJECTTRIALS|EEGPLOT", tag="vistype", value=2),
            ControlSpec("spacer"),
            ControlSpec("text", "Display previous rejection marks"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="superpose", value=True),
            ControlSpec("text", "Reject marked trial(s)"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="reject", value=False),
        ),
    )


def _run_gui(EEG: dict[str, Any], icacomp: int, renderer: Any | None) -> tuple[Any, ...] | None:
    result = inputgui(pop_jointprob_dialog_spec(EEG, icacomp), renderer=renderer)
    if result is None:
        return None
    threshold_default = "3" if int(bool(icacomp)) else "5"
    return (
        result.get("elecrange", ""),
        result.get("locthresh", threshold_default),
        result.get("globthresh", threshold_default),
        int(bool(result.get("superpose", True))),
        int(bool(result.get("reject", False))),
        vistype_from_gui(result.get("vistype", 2)),
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
    *,
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], list[float], list[float], list[int], str]:
    return run_epoched_rejection(
        EEG,
        icacomp,
        elecrange,
        locthresh,
        globthresh,
        superpose,
        reject,
        vistype,
        marks_fn=jointprob_marks,
        kind="rejjp",
        stats_local_field="jpE",
        stats_global_field="jp",
        stats_local_field_ica="icajpE",
        stats_global_field_ica="icajp",
        error_message="pop_jointprob requires epoched data",
        command_fn=lambda normalized_elecrange: _history_command(
            icacomp, normalized_elecrange, locthresh, globthresh, superpose, reject, vistype, 0
        ),
        display=display,
        command_callback=command_callback,
        show=show,
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
        int(superpose),
        int(bool(reject)),
        int(vistype),
        [],
        int(bool(plotflag)),
    ]
    return (
        "EEG = pop_jointprob(EEG, "
        + ", ".join(format_history_value(arg, cell_for_sequence=None) for arg in args)
        + ");"
    )

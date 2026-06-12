"""Reject epochs with linear-trend artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._eegplot_rejection import run_epoched_mark_rejection
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import (
    parse_numeric_sequence,
    trend_marks,
)


def pop_rejtrend(
    EEG: dict[str, Any] | list[dict[str, Any]],
    icacomp: int | bool = 1,
    elecrange: Any = None,
    winsize: Any = None,
    minslope: Any = 0.5,
    minstd: Any = 0.3,
    superpose: int | bool = 0,
    reject: int | bool = 1,
    calldisp: int | bool = 0,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
):
    """Mark or reject epochs containing line-like trends."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = elecrange is None and winsize is None
    if isinstance(EEG, list):
        if gui:
            result = _run_gui(EEG[0], int(bool(icacomp)), renderer=renderer)
            if result is None:
                return (EEG, "") if return_com else EEG
            elecrange, winsize, minslope, minstd, superpose, reject = result
        outputs = [
            _apply_one(
                dataset,
                icacomp,
                elecrange,
                winsize,
                minslope,
                minstd,
                superpose,
                reject,
                display=False,
            )[0]
            for dataset in EEG
        ]
        command = _history_command(icacomp, elecrange, winsize, minslope, minstd, superpose, reject)
        return (outputs, command) if return_com else outputs
    if gui:
        result = _run_gui(EEG, int(bool(icacomp)), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        elecrange, winsize, minslope, minstd, superpose, reject = result
    out, command = _apply_one(
        EEG,
        icacomp,
        elecrange,
        winsize,
        minslope,
        minstd,
        superpose,
        reject,
        display=bool(gui or calldisp),
        command_callback=command_callback,
        show=show,
    )
    return (out, command) if return_com else out


def pop_rejtrend_dialog_spec(EEG: dict[str, Any], icacomp: int | bool = 1) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rejtrend``."""
    is_data = int(bool(icacomp))
    rows = int(EEG.get("nbchan", 0) or 0) if is_data else int(np.asarray(EEG.get("icaweights", [])).shape[0])
    title = "Data trend rejection -- pop_rejtrend()" if is_data else "Trend rejection in component(s) -- pop_rejtrend()"
    row_label = "Electrode (indices; Ex: 2 6:8 10):" if is_data else "Component (indices; Ex: 2 6:8 10):"
    return DialogSpec(
        title=title,
        function_name="pop_rejtrend",
        eeglab_source="functions/popfunc/pop_rejtrend.m",
        help_text="pophelp('pop_rejtrend')",
        size=(600, 304),
        geometry=((1, 0.1, 0.75),) * 4 + (1,) + ((1, 0.22, 0.85),) * 2,
        controls=(
            ControlSpec("text", row_label),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="elecrange", value=f"1:{rows}"),
            ControlSpec("text", "Slope window width (in points)"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="winsize", value=str(int(EEG.get("pnts", 1) or 1))),
            ControlSpec("text", "Maximum slope to allow"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="minslope", value="0.5"),
            ControlSpec("text", "R-square limit to allow ([0:1])"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="minstd", value="0.3"),
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
    result = inputgui(pop_rejtrend_dialog_spec(EEG, icacomp), renderer=renderer)
    if result is None:
        return None
    return (
        result.get("elecrange", ""),
        result.get("winsize", ""),
        result.get("minslope", "0.5"),
        result.get("minstd", "0.3"),
        int(bool(result.get("superpose", False))),
        int(bool(result.get("reject", False))),
    )


def _apply_one(
    EEG: dict[str, Any],
    icacomp: int | bool,
    elecrange: Any,
    winsize: Any,
    minslope: Any,
    minstd: Any,
    superpose: int | bool,
    reject: int | bool,
    *,
    display: bool = False,
    command_callback: Any | None = None,
    show: bool = True,
) -> tuple[dict[str, Any], str]:
    def _marks(_out: dict[str, Any], data: np.ndarray, normalized_elecrange: list[int]):
        normalized = {
            "winsize": int(parse_numeric_sequence(winsize if winsize is not None else [data.shape[1]], dtype=float)[0]),
            "minslope": float(parse_numeric_sequence(minslope, dtype=float)[0]),
            "minstd": float(parse_numeric_sequence(minstd, dtype=float)[0]),
        }
        marks, marks_e = trend_marks(
            data,
            normalized_elecrange,
            normalized["winsize"],
            normalized["minslope"],
            normalized["minstd"],
        )
        return marks, marks_e, normalized

    def _command(normalized_elecrange: list[int], normalized: dict[str, Any]) -> str:
        return _history_command(
            icacomp,
            normalized_elecrange,
            normalized["winsize"],
            normalized["minslope"],
            normalized["minstd"],
            superpose,
            reject,
        )

    out, _rejected, command, _normalized = run_epoched_mark_rejection(
        EEG,
        icacomp,
        elecrange,
        superpose,
        reject,
        marks_fn=_marks,
        kind="rejconst",
        error_message="pop_rejtrend requires epoched data",
        command_fn=_command,
        display=display,
        command_callback=command_callback,
        show=show,
    )
    return out, command


def _history_command(
    icacomp: int | bool,
    elecrange: Any,
    winsize: Any,
    minslope: Any,
    minstd: Any,
    superpose: int | bool,
    reject: int | bool,
) -> str:
    args = [
        int(bool(icacomp)),
        parse_numeric_sequence(elecrange, dtype=int),
        int(parse_numeric_sequence(winsize, dtype=float)[0]),
        float(parse_numeric_sequence(minslope, dtype=float)[0]),
        float(parse_numeric_sequence(minstd, dtype=float)[0]),
        int(superpose),
        int(bool(reject)),
    ]
    return (
        "EEG = pop_rejtrend(EEG, " + ", ".join(format_history_value(arg, cell_for_sequence=None) for arg in args) + ");"
    )

"""Inspect and mark ICA components for rejection."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc._rejection import component_rejection_flags, copy_eeg, one_based_indices
from eegprep.functions.popfunc.pop_topoplot import pop_topoplot


PLOTS_PER_FIGURE = 35


def pop_selectcomps(
    EEG: dict[str, Any],
    compnum: Any = None,
    fig: Any = None,
    *,
    reject: Any = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: bool = True,
    return_com: bool = False,
):
    """Inspect components and optionally mark them in ``EEG.reject.gcompreject``.

    EEGPrep keeps the non-browser data workflow explicit: component maps can be
    plotted, and callers can pass ``reject=[...]`` to mark components for later
    ``pop_subcomp`` removal.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = compnum is None and reject is None
    if gui:
        result = inputgui(pop_selectcomps_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        compnum = result.get("compnum", "")
        reject = result.get("reject", "")
    out = copy_eeg(EEG)
    n_components = _component_count(out)
    if n_components == 0:
        raise ValueError("ICA decomposition is required")
    compnum = one_based_indices(compnum, limit=n_components, default_all=True)
    reject_values = one_based_indices(reject, limit=n_components, default_all=False)
    if reject_values:
        flags = component_rejection_flags(out, n_components, create=False).astype(int)
        flags[np.asarray(reject_values, dtype=int) - 1] = 1
        if not isinstance(out.get("reject"), dict):
            out["reject"] = {}
        out["reject"]["gcompreject"] = flags
    if plot and compnum:
        pop_topoplot(
            out, 0, compnum[:PLOTS_PER_FIGURE], "Reject components by map - pop_selectcomps()", [], 0, gui=False
        )
    command = _history_command(compnum, reject_values)
    return (out, command) if return_com else out


def pop_selectcomps_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like component selection prompt."""
    n_components = _component_count(EEG)
    return DialogSpec(
        title="Reject comp. by map -- pop_selectcomps",
        function_name="pop_selectcomps",
        eeglab_source="functions/popfunc/pop_selectcomps.m",
        help_text="pophelp('pop_selectcomps')",
        size=(530, 266),
        geometry=((1, 1), 1, 1, (1, 1)),
        controls=(
            ControlSpec("text", "Components to plot:"),
            ControlSpec("edit", tag="compnum", value=f"1:{n_components}"),
            ControlSpec("spacer"),
            ControlSpec(
                "text",
                'Note: inspect component maps, then use "Components to mark" to label them for rejection. '
                'To actually reject labelled components use "Tools > Remove components from data".',
            ),
            ControlSpec("text", "Components to mark for rejection:"),
            ControlSpec("edit", tag="reject", value=_flagged_component_text(EEG)),
        ),
        known_differences=("EEGPrep does not use an EEGPlot-style scrolling figure for component toggles.",),
    )


def _history_command(compnum: list[int], reject: list[int]) -> str:
    args = [format_history_value(compnum, cell_for_sequence=None)]
    if reject:
        args.append(f"reject={format_history_value(reject, cell_for_sequence=None)}")
    return "EEG = pop_selectcomps(EEG, " + ", ".join(args) + ");"


def _component_count(EEG: dict[str, Any]) -> int:
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def _flagged_component_text(EEG: dict[str, Any]) -> str:
    flags = np.asarray((EEG.get("reject") or {}).get("gcompreject", []), dtype=bool).ravel()
    return " ".join(str(index + 1) for index, flag in enumerate(flags) if flag)

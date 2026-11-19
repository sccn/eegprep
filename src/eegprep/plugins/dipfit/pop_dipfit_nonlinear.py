"""DIPFIT pop_dipfit_nonlinear workflow and preserved GUI specification."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.plugins.dipfit._utils import (
    component_count,
    copy_eeg,
    dipfit_history,
    dipfit_record,
    ensure_dipfit_settings,
    one_based_indices,
    normalize_model_list,
)

from eegprep.plugins.dipfit.dipfit_gridsearch import empty_model, ensure_model_list
from eegprep.plugins.dipfit.dipfit_nonlinear import dipfit_nonlinear as run_nonlinear


def pop_dipfit_nonlinear(
    EEG: dict[str, Any] | None = None,
    component: Any = None,
    *args: Any,
    nonlinear: str | bool = "yes",
    symmetry: str | None = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Run standalone nonlinear dipole refinement for one component."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_dipfit_settings(EEG)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if "component" in options and component is None:
        component = options.pop("component")
    nonlinear = options.pop("nonlinear", nonlinear)
    symmetry = options.pop("symmetry", symmetry)
    if gui is None:
        gui = component is None and not options
    working = EEG
    if gui:
        result = inputgui(pop_dipfit_nonlinear_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        working, component = _apply_nonlinear_gui_model(EEG, result)
        if result.get("dip2sel") and result.get("dip2sym"):
            symmetry = "x" if str(dipfit_record(EEG).get("coordformat", "")).lower() == "mni" else "y"
    component_index = _single_component(component, working)
    out = run_nonlinear(working, component=component_index, nonlinear=nonlinear, symmetry=symmetry)
    command = dipfit_history(
        "pop_dipfit_nonlinear",
        component=component_index,
        nonlinear=nonlinear,
        symmetry=symmetry or "",
    )
    return (out, command) if return_com else out


def pop_dipfit_nonlinear_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return a compact version of EEGLAB's manual fine-fit dialog."""
    component, model = _current_dialog_model(EEG)
    first_selected = 1 in _dialog_selected_dipoles(model)
    second_selected = 2 in _dialog_selected_dipoles(model)
    return DialogSpec(
        title="Manual dipole fit -- pop_dipfit_nonlinear()",
        function_name="pop_dipfit_nonlinear",
        eeglab_source="plugins/dipfit/pop_dipfit_nonlinear.m",
        help_text="pophelp('pop_dipfit_nonlinear')",
        size=(650, 330),
        geometry=(
            (0.8, 0.5, 0.8, 1, 1),
            (1,),
            (0.7, 0.7, 2, 2, 1),
            (0.7, 0.5, 0.2, 2, 2, 1),
            (0.7, 0.5, 0.2, 2, 2, 1),
            (1,),
            (1, 1, 1),
        ),
        controls=(
            ControlSpec("text", "Component to fit"),
            ControlSpec("edit", tag="component", value=str(component)),
            ControlSpec("pushbutton", "Plot map", tag="plotmap", enabled=False),
            ControlSpec("text", "Residual variance = "),
            ControlSpec("text", _format_dialog_rv(model), tag="relvar"),
            ControlSpec("spacer"),
            ControlSpec("text", "dipole"),
            ControlSpec("text", "fit"),
            ControlSpec("text", "position"),
            ControlSpec("text", "moment"),
            ControlSpec("spacer"),
            ControlSpec("text", "1", tag="dip1"),
            ControlSpec("checkbox", tag="dip1sel", value=first_selected),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="dip1pos", value=_format_dialog_triplet(model.get("posxyz"), 0)),
            ControlSpec("edit", tag="dip1mom", value=_format_dialog_triplet(model.get("momxyz"), 0)),
            ControlSpec("pushbutton", "Flip (in|out)", tag="flip1", enabled=False),
            ControlSpec("text", "#2", tag="dip2"),
            ControlSpec("checkbox", tag="dip2sel", value=second_selected),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="dip2pos", value=_format_dialog_triplet(model.get("posxyz"), 1)),
            ControlSpec("edit", tag="dip2mom", value=_format_dialog_triplet(model.get("momxyz"), 1)),
            ControlSpec("pushbutton", "Flip (in|out)", tag="flip2", enabled=False),
            ControlSpec("checkbox", "Symmetry constrain for dipole #2", tag="dip2sym", value=True),
            ControlSpec("pushbutton", "Fit dipole(s)' position & moment", tag="fitposition", enabled=False),
            ControlSpec("pushbutton", "OR fit only dipole(s)' moment", tag="fitmoment", enabled=False),
            ControlSpec("pushbutton", "Plot dipole(s)", tag="plotdip", enabled=False),
        ),
        known_differences=(
            "EEGPrep OK applies the displayed values and runs a spherical nonlinear fit; individual MATLAB callbacks are not mirrored.",
        ),
    )


def _current_dialog_model(EEG: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    count = _component_count_or_one(EEG)
    current = dipfit_record(EEG).get("current", 1) or 1
    try:
        component = one_based_indices([current], limit=count, default_all=False)[0]
    except ValueError:
        component = 1
    models = normalize_model_list(EEG)
    if component <= len(models):
        return component, models[component - 1]
    return component, empty_model(component)


def _dialog_selected_dipoles(model: dict[str, Any]) -> list[int]:
    selected = model.get("select", model.get("active", [1]))
    try:
        indices = numeric_vector(selected, dtype=int).tolist()
    except ValueError:
        return [1]
    return [int(index) for index in indices if int(index) in (1, 2)] or [1]


def _format_dialog_triplet(value: Any, row_index: int) -> str:
    try:
        values = numeric_vector(value, dtype=float)
    except ValueError:
        values = np.asarray([], dtype=float)
    if values.size < 3:
        return "0 0 0"
    rows = values.reshape(-1, 3)
    if row_index >= rows.shape[0]:
        return "0 0 0"
    row = rows[row_index]
    return " ".join(f"{float(item):g}" for item in row)


def _format_dialog_rv(model: dict[str, Any]) -> str:
    try:
        has_position = numeric_vector(model.get("posxyz", []), dtype=float).size >= 3
    except ValueError:
        has_position = False
    if not has_position:
        return "not fit"
    try:
        rv = float(np.asarray(model.get("rv", []), dtype=float).ravel()[0])
    except (IndexError, TypeError, ValueError):
        return "not fit"
    if not np.isfinite(rv):
        return "not fit"
    return f"{rv * 100:g}%"


def _single_component(value: Any, EEG: dict[str, Any]) -> int:
    if value is None or value == "":
        current = dipfit_record(EEG).get("current", 1)
        value = current if current else 1
    return one_based_indices([value], limit=component_count(EEG), default_all=False)[0]


def _apply_nonlinear_gui_model(EEG: dict[str, Any], result: dict[str, Any]) -> tuple[dict[str, Any], int]:
    out = copy_eeg(EEG)
    component = _single_component(result.get("component", 1), out)
    models = ensure_model_list(out, component_count(out))
    positions = []
    moments = []
    selected = []
    for index, prefix in enumerate(("dip1", "dip2"), start=1):
        if result.get(f"{prefix}sel", index == 1):
            selected.append(index)
        positions.append(numeric_vector(result.get(f"{prefix}pos", "0 0 0"), dtype=float).tolist())
        moments.append(numeric_vector(result.get(f"{prefix}mom", "0 0 0"), dtype=float).tolist())
    models[component - 1].update(
        {"posxyz": positions, "momxyz": moments, "select": selected or [1], "component": component}
    )
    out["dipfit"]["current"] = component
    return out, component


def _component_count_or_one(EEG: dict[str, Any]) -> int:
    return max(component_count(EEG), 1)

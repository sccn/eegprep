"""DIPFIT pop_multifit workflow and preserved GUI specification."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.plugins.dipfit._utils import (
    component_count,
    copy_eeg,
    dipfit_history,
    dipfit_record,
    ensure_dipfit_settings,
    ensure_ica,
    matlab_plot_options,
    one_based_indices,
    normalize_model_list,
)

from eegprep.plugins.dipfit.dipfit_gridsearch import dipfit_gridsearch as run_gridsearch, empty_model, ensure_model_list
from eegprep.plugins.dipfit.dipfit_nonlinear import dipfit_nonlinear as run_nonlinear
from eegprep.plugins.dipfit.dipfit_reject import dipfit_reject
from eegprep.plugins.dipfit.eeg_compatlas import eeg_compatlas
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot


def pop_multifit(
    EEG: dict[str, Any] | None = None,
    comps: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Fit multiple ICA components using grid search and nonlinear refinement."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_ica(EEG)
    ensure_dipfit_settings(EEG)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = comps is None and not options
    if gui:
        result = inputgui(pop_multifit_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        comps = result.get("comps", "")
        options.update(_multifit_options_from_result(result))
    count = component_count(EEG)
    components = one_based_indices(comps, limit=count, default_all=True)
    threshold = float(np.asarray(options.get("threshold", 40), dtype=float).ravel()[0])
    dipoles = int(float(np.asarray(options.get("dipoles", 1), dtype=float).ravel()[0]))
    if not 0 <= threshold <= 100 or dipoles not in (1, 2):
        raise ValueError("threshold must be in 0..100 and dipoles must be 1 or 2")
    out = _ensure_gridsearch_starting_positions(EEG, components)
    if dipoles == 2:
        out = _make_bilateral_starting_models(out, components)
    symmetry = "x" if str(dipfit_record(out).get("coordformat", "")).lower() == "mni" else "y"
    for component_index in components:
        try:
            out = run_nonlinear(
                out,
                component=component_index,
                symmetry=symmetry if dipoles == 2 else None,
            )
        except (ValueError, np.linalg.LinAlgError):
            models = ensure_model_list(out, count)
            models[component_index - 1] = empty_model(component_index)
    models = dipfit_reject(out["dipfit"].get("model", []), threshold / 100)
    if (
        str(options.get("rmout", "off")).lower() == "on"
        and str(dipfit_record(out).get("coordformat", "")).lower() == "spherical"
    ):
        for index, model in enumerate(models):
            positions = np.asarray(model.get("posxyz", []), dtype=float).reshape(-1, 3)
            if np.any(np.linalg.norm(positions, axis=1) > 85.0):
                models[index] = empty_model(index + 1)
    out["dipfit"]["model"] = models
    record = dipfit_record(out)
    coordformat = str(record.get("coordformat", "")).lower()
    if coordformat in {"mni", "spherical"} or np.asarray(record.get("coord_transform", []), dtype=float).size:
        eeg_compatlas(out)
    command = dipfit_history(
        "pop_multifit",
        components,
        threshold=threshold,
        rmout=options.get("rmout", "off"),
        dipoles=dipoles,
        dipplot=options.get("dipplot", "off"),
        plotopt=options.get("plotopt", ""),
    )
    if str(options.get("dipplot", "off")).lower() == "on":
        plot_options = matlab_plot_options(options.get("plotopt", ""))
        pop_dipplot(out, components, **plot_options)
    return (out, command) if return_com else out


def pop_multifit_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like multi-component autofit dialog."""
    return DialogSpec(
        title="Fit multiple ICA components -- pop_multifit()",
        function_name="pop_multifit",
        eeglab_source="plugins/dipfit/pop_multifit.m",
        help_text="pophelp('pop_multifit')",
        size=(560, 300),
        geometry=((1.91, 1), (1.91, 1), (1,), (1,), (1,), (0.5, 1.5, 1.5, 0.8)),
        controls=(
            ControlSpec("text", "Component indices (default:all)"),
            ControlSpec("edit", tag="comps", value=""),
            ControlSpec("text", "Rejection threshold RV (%)"),
            ControlSpec("edit", tag="threshold", value="100"),
            ControlSpec("checkbox", "Remove dipoles outside the head", tag="rmout", value=False),
            ControlSpec("checkbox", "Fit bilateral dipoles (check)", tag="dipoles", value=False),
            ControlSpec(
                "checkbox",
                "Plot resulting dipoles using dipplot (check)",
                tag="dipplot",
                value=False,
                callback=CallbackSpec(
                    "toggle_enabled",
                    params={"source": "dipplot", "targets": ["plot_label", "plotopt", "plot_help"]},
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "plotting options", tag="plot_label", enabled=False),
            ControlSpec("edit", tag="plotopt", value="'normlen' 'on'", enabled=False),
            ControlSpec("pushbutton", "Help", tag="plot_help", enabled=False),
        ),
        known_differences=(
            "standardBEM autofit uses the packaged dipoli leadfield. standardBESA autofit uses the four-shell BESA leadfield.",
        ),
    )


def _multifit_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": result.get("threshold", "100"),
        "rmout": "on" if result.get("rmout", False) else "off",
        "dipoles": 2 if result.get("dipoles", False) else 1,
        "dipplot": "on" if result.get("dipplot", False) else "off",
        "plotopt": result.get("plotopt", ""),
    }


def _ensure_gridsearch_starting_positions(EEG: dict[str, Any], components: list[int]) -> dict[str, Any]:
    models = normalize_model_list(EEG)
    if all(
        component <= len(models) and np.asarray(models[component - 1].get("posxyz", [])).size
        for component in components
    ):
        return EEG
    return run_gridsearch(EEG, component=components, reject=100)


def _make_bilateral_starting_models(EEG: dict[str, Any], components: list[int]) -> dict[str, Any]:
    out = copy_eeg(EEG)
    models = ensure_model_list(out, component_count(out))
    coordformat = str(dipfit_record(out).get("coordformat", "")).lower()
    axis = 0 if coordformat == "mni" else 1
    for component in components:
        model = models[component - 1]
        first = np.asarray(model.get("posxyz", [0.0, -40.0, 30.0]), dtype=float).reshape(-1, 3)[0]
        second = first.copy()
        first[axis] = -40.0
        second[axis] = 40.0
        moment = np.asarray(model.get("momxyz", [0.0, 0.0, 0.0]), dtype=float).reshape(-1, 3)[0]
        model.update(
            {
                "posxyz": [first.tolist(), second.tolist()],
                "momxyz": [moment.tolist(), moment.tolist()],
                "active": [1, 2],
                "select": [1, 2],
            }
        )
    out["dipfit"]["model"] = models
    return out

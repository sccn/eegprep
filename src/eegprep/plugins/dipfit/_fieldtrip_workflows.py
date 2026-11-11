"""EEGLAB-style DIPFIT workflows."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.io import loadmat

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.plugins.dipfit._fitting import (
    DEFAULT_HEAD_RADIUS_MM,
    dipfit_gridsearch as run_gridsearch,
    dipfit_nonlinear as run_nonlinear,
    dipfit_reject,
    empty_model,
    ensure_model_list,
    loreta_key_source_points,
    parse_grid_values,
    remove_outside_head,
    source_model_from_points,
)
from eegprep.plugins.dipfit._utils import (
    DIPFITUnavailableError,
    component_count,
    copy_eeg,
    dipfit_history,
    dipfit_record,
    ensure_channel_locations,
    fill_missing_besa_volume,
    ensure_dipfit_settings,
    ensure_ica,
    matlab_plot_options,
    one_based_indices,
    unavailable,
)
from eegprep.plugins.dipfit.pop_dipplot import pop_dipplot


def pop_dipfit_headmodel(
    EEG: dict[str, Any] | None = None,
    mriFile: str | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Validate headmodel inputs, then fail clearly until FieldTrip is ported."""
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = mriFile is None and not options
    if gui:
        result = inputgui(pop_dipfit_headmodel_dialog_spec(EEG, mriFile or ""), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        mriFile = result.get("mrifile", "")
        options.update(_headmodel_options_from_result(result))
    if not mriFile:
        raise ValueError("MRI file is required to create a DIPFIT head model")
    unavailable("pop_dipfit_headmodel")


def pop_dipfit_gridsearch(
    EEG: dict[str, Any] | None = None,
    select: Any = None,
    xgrid: Any = None,
    ygrid: Any = None,
    zgrid: Any = None,
    reject: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Run a standalone spherical coarse grid search for ICA components."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    ensure_dipfit_settings(EEG, caller="pop_dipfit_gridsearch")
    if gui is None:
        gui = select is None
    if gui:
        result = inputgui(pop_dipfit_gridsearch_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        select = result.get("select", "")
        xgrid = result.get("xgrid", "")
        ygrid = result.get("ygrid", "")
        zgrid = result.get("zgrid", "")
        reject = result.get("reject", "")
    count = component_count(EEG)
    components = one_based_indices(select, limit=count, default_all=True)
    reject_value = _blank_to_none(reject)
    out = run_gridsearch(EEG, component=components, xgrid=xgrid, ygrid=ygrid, zgrid=zgrid, reject=reject_value)
    command = dipfit_history(
        "pop_dipfit_gridsearch",
        components,
        _history_grid(xgrid, np.linspace(-DEFAULT_HEAD_RADIUS_MM, DEFAULT_HEAD_RADIUS_MM, 11)),
        _history_grid(ygrid, np.linspace(-DEFAULT_HEAD_RADIUS_MM, DEFAULT_HEAD_RADIUS_MM, 11)),
        _history_grid(zgrid, np.linspace(0.0, DEFAULT_HEAD_RADIUS_MM, 6)),
        None if reject_value is None else float(reject_value),
    )
    return (out, command) if return_com else out


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
    ensure_dipfit_settings(EEG, caller="pop_dipfit_nonlinear")
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
    ensure_dipfit_settings(EEG, caller="pop_multifit")
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
    models = dipfit_reject(out["dipfit"].get("model", []), threshold)
    if str(options.get("rmout", "off")).lower() == "on":
        models, _removed = remove_outside_head(models)
    out["dipfit"]["model"] = models
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


def pop_leadfield(
    EEG: dict[str, Any] | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Compute a standalone spherical leadfield for explicit source points."""
    if EEG is None:
        return (None, "") if return_com else None
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        result = inputgui(pop_leadfield_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(_leadfield_options_from_result(result))
    if not options.get("sourcemodel"):
        raise ValueError(
            "pop_leadfield: sourcemodel is empty. "
            "Pass source points, a file, or sourcemodel='loreta-key' for EEGPrep's in-head grid."
        )
    points = _source_points_from_option(options.get("sourcemodel"))
    out = copy_eeg(EEG)
    ensure_dipfit_settings(out, caller="pop_leadfield")
    fill_missing_besa_volume(out)
    out["dipfit"]["sourcemodel"] = source_model_from_points(out, points)
    out["dipfit"]["sourcemodel"]["file"] = options.get("sourcemodel")
    if options.get("sourcemodel2mni") not in (None, ""):
        out["dipfit"]["sourcemodel"]["coordtransform"] = numeric_vector(options["sourcemodel2mni"]).tolist()
    command = dipfit_history(
        "pop_leadfield",
        sourcemodel=options.get("sourcemodel"),
        sourcemodel2mni=options.get("sourcemodel2mni", ""),
        downsample=options.get("downsample", ""),
    )
    return (out, command) if return_com else out


def pop_dipfit_loreta(
    EEG: dict[str, Any] | None = None,
    select: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Validate LORETA prerequisites, then fail clearly until a backend exists."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    dipfit = ensure_dipfit_settings(EEG, caller="pop_dipfit_loreta")
    if not dipfit.get("sourcemodel"):
        raise ValueError(
            "pop_dipfit_loreta: no leadfield is stored in EEG.dipfit.sourcemodel. "
            "Run pop_leadfield first. The LORETA-KEY choice builds EEGPrep's in-head grid."
        )
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = select is None and not options
    if gui:
        result = inputgui(pop_dipfit_loreta_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        select = result.get("select", "")
        options.update(_loreta_options_from_result(result))
    unavailable("pop_dipfit_loreta")


def pop_dipfit_batch(*args: Any, **kwargs: Any):
    """Deprecated EEGLAB alias for :func:`pop_dipfit_gridsearch`."""
    return pop_dipfit_gridsearch(*args, **kwargs)


def pop_dipfit_manual(*args: Any, **kwargs: Any):
    """Deprecated EEGLAB alias for :func:`pop_dipfit_nonlinear`."""
    return pop_dipfit_nonlinear(*args, **kwargs)


def pop_dipfit_headmodel_dialog_spec(EEG: dict[str, Any], mri_file: str = "") -> DialogSpec:
    """Return the EEGLAB-like headmodel dialog after MRI selection."""
    return DialogSpec(
        title="Create headmodel from MRI - pop_dipfit_headmodel",
        function_name="pop_dipfit_headmodel",
        eeglab_source="plugins/dipfit/pop_dipfit_headmodel.m",
        help_text="pophelp('pop_dipfit_headmodel')",
        size=(650, 280),
        geometry=((3, 0.5, 0.5), (2, 1.5, 0.5), (2, 1.5, 0.5), (2, 1.5, 0.5), (2, 1.5, 0.5), (2, 1.5, 0.5)),
        controls=(
            ControlSpec("text", "Parameters to calculate head model from MRI", font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("text", "plot"),
            ControlSpec("text", "MRI file"),
            ControlSpec("edit", tag="mrifile", value=mri_file),
            ControlSpec("spacer"),
            ControlSpec("text", "Nasion X, Y, Z (MRI voxels space)"),
            ControlSpec("edit", tag="nasion", value=""),
            ControlSpec("checkbox", tag="plotnasion", value=False),
            ControlSpec("text", "Left ear (LPA) X, Y, Z (MRI voxels space)"),
            ControlSpec("edit", tag="lpa", value=""),
            ControlSpec("checkbox", tag="plotlpa", value=False),
            ControlSpec("text", "Right ear (RPA) X, Y, Z (MRI voxels space)"),
            ControlSpec("edit", tag="rpa", value=""),
            ControlSpec("checkbox", tag="plotrpa", value=False),
            ControlSpec("text", "Select EEG (3 surfaces) or MEG (1 surface)"),
            ControlSpec("popupmenu", "EEG|MEG", tag="datatype", value=1),
            ControlSpec("checkbox", tag="plotsurf", value=False),
        ),
        known_differences=(
            "EEGPrep collects the EEGLAB parameters but cannot yet call FieldTrip MRI/headmodel routines.",
        ),
    )


def pop_dipfit_gridsearch_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like coarse dipole fit dialog."""
    ncomp = _component_count_or_one(EEG)
    return DialogSpec(
        title="Batch dipole fit -- pop_dipfit_gridsearch()",
        function_name="pop_dipfit_gridsearch",
        eeglab_source="plugins/dipfit/pop_dipfit_gridsearch.m",
        help_text="pophelp('pop_dipfit_gridsearch')",
        size=(520, 260),
        geometry=((1, 1), (1, 1), (1, 1), (1, 1), (1, 1)),
        controls=(
            ControlSpec("text", "Component(s) (not faster if few comp.)"),
            ControlSpec("edit", tag="select", value=f"1:{ncomp}"),
            ControlSpec("text", "Grid in X-direction"),
            ControlSpec("edit", tag="xgrid", value="linspace(-85,85,24)"),
            ControlSpec("text", "Grid in Y-direction"),
            ControlSpec("edit", tag="ygrid", value="linspace(-85,85,24)"),
            ControlSpec("text", "Grid in Z-direction"),
            ControlSpec("edit", tag="zgrid", value="linspace(0,85,12)"),
            ControlSpec("text", "Rejection threshold RV(%)"),
            ControlSpec("edit", tag="reject", value="40"),
        ),
        known_differences=(
            "EEGPrep uses a deterministic spherical leadfield backend; FieldTrip BEM fitting is still not bundled.",
        ),
    )


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
            "EEGPrep autofit uses a native spherical leadfield backend; FieldTrip BEM fitting is still not bundled.",
        ),
    )


def pop_leadfield_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like leadfield source-model dialog."""
    return DialogSpec(
        title="Compute ROI activity",
        function_name="pop_leadfield",
        eeglab_source="plugins/dipfit/pop_leadfield.m",
        help_text="pophelp('pop_leadfield')",
        size=(650, 230),
        geometry=((1,), (1,), (0.1, 0.8, 0.8, 0.2), (0.1, 0.8, 0.8, 0.2), (0.1, 0.8, 0.2, 0.8)),
        controls=(
            ControlSpec("text", "Choose source model for Leadfield matrix", font_weight="bold"),
            ControlSpec(
                "popupmenu",
                "Surface source model: Colin27 (with Desikan-Kilianny atlas)|Surface source model: Use Brainstorm ICBM152 (with Desikan-Kilianny atlas)|Volumetric source model: LORETA-KEY|Volumetric source model: AFNI with TTatlas+tlrc atlas (Fieldtrip)|Custom source model",
                tag="source_choice",
                value=3,
                callback=CallbackSpec(
                    "choice_text",
                    params={
                        "source": "source_choice",
                        "target": "sourcemodel",
                        "browse": "source_browse",
                        "texts": ["", "", "loreta-key", "", ""],
                    },
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "File name"),
            ControlSpec("edit", tag="sourcemodel", value="loreta-key"),
            ControlSpec(
                "pushbutton",
                "...",
                tag="source_browse",
                enabled=False,
                callback=CallbackSpec("select_file", params={"button": "source_browse", "target": "sourcemodel"}),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Transformation to MNI (if any)"),
            ControlSpec("edit", tag="sourcemodel2mni", value=""),
            ControlSpec("pushbutton", "...", tag="transform_browse", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Downsampling (AFNI only)"),
            ControlSpec("edit", tag="downsample", value="4", enabled=False),
            ControlSpec("spacer"),
        ),
        known_differences=(
            "The LORETA-KEY choice builds EEGPrep's 20 mm in-head grid and a spherical leadfield.",
            "Colin27, ICBM152, and AFNI atlas files are not bundled.",
        ),
    )


def pop_dipfit_loreta_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like LORETA component-localization dialog."""
    return DialogSpec(
        title="Localization of ICA components using eLoreta -- pop_dipfit_loreta()",
        function_name="pop_dipfit_loreta",
        eeglab_source="plugins/dipfit/pop_dipfit_loreta.m",
        help_text="pophelp('pop_dipfit_loreta')",
        size=(620, 250),
        geometry=((1,), (1,), (1,), (1,), (1,), (1,)),
        controls=(
            ControlSpec("text", "Enter indices of components \n(one figure generated per component)"),
            ControlSpec("edit", tag="select", value="1"),
            ControlSpec("text", "ft_sourceanalysis parameters"),
            ControlSpec("edit", tag="ft_sourceanalysis_params", value="'method', 'eloreta'"),
            ControlSpec("text", "ft_sourceplot parameters"),
            ControlSpec("edit", tag="ft_sourceplot_params", value="'method', 'slice'"),
        ),
        known_differences=(
            "LORETA source analysis requires FieldTrip sourceanalysis/sourceplot and is not bundled yet.",
        ),
    )


def _headmodel_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"datatype": "meg" if int(result.get("datatype", 1)) == 2 else "eeg"}
    for key in ("nasion", "lpa", "rpa"):
        value = str(result.get(key, "")).strip()
        if value:
            options[key] = value
    if result.get("plotsurf", False):
        options["plotmesh"] = "scalp"
    fiducials = [
        name for name, key in (("nasion", "plotnasion"), ("lpa", "plotlpa"), ("rpa", "plotrpa")) if result.get(key)
    ]
    if fiducials:
        options["plotfiducial"] = fiducials
    return options


def _multifit_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": result.get("threshold", "100"),
        "rmout": "on" if result.get("rmout", False) else "off",
        "dipoles": 2 if result.get("dipoles", False) else 1,
        "dipplot": "on" if result.get("dipplot", False) else "off",
        "plotopt": result.get("plotopt", ""),
    }


def _leadfield_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    choice = int(result.get("source_choice", 3) or 3)
    sourcemodel = str(result.get("sourcemodel") or "").strip()
    if not sourcemodel and choice == 3:
        sourcemodel = "loreta-key"
    if not sourcemodel:
        raise ValueError(
            "pop_leadfield: that source model file is not bundled. "
            "Leave the LORETA-KEY choice selected to use EEGPrep's in-head grid, "
            "or pass source points in sourcemodel."
        )
    return {
        "sourcemodel": sourcemodel,
        "sourcemodel2mni": result.get("sourcemodel2mni", ""),
        "downsample": result.get("downsample", ""),
    }


def _loreta_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ft_sourceanalysis_params": result.get("ft_sourceanalysis_params", ""),
        "ft_sourceplot_params": result.get("ft_sourceplot_params", ""),
    }


def _component_count_or_one(EEG: dict[str, Any]) -> int:
    return max(component_count(EEG), 1)


def _current_dialog_model(EEG: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    count = _component_count_or_one(EEG)
    current = dipfit_record(EEG).get("current", 1) or 1
    try:
        component = one_based_indices([current], limit=count, default_all=False)[0]
    except ValueError:
        component = 1
    models = normalize_existing_models(EEG)
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


def _ensure_gridsearch_starting_positions(EEG: dict[str, Any], components: list[int]) -> dict[str, Any]:
    models = normalize_existing_models(EEG)
    if all(
        component <= len(models) and np.asarray(models[component - 1].get("posxyz", [])).size
        for component in components
    ):
        return EEG
    return run_gridsearch(EEG, component=components, reject=100)


def normalize_existing_models(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    models = dipfit_record(EEG).get("model", [])
    if isinstance(models, dict):
        return [models]
    if isinstance(models, np.ndarray):
        models = models.tolist()
    return [dict(model) for model in models] if isinstance(models, list) else []


def _make_bilateral_starting_models(EEG: dict[str, Any], components: list[int]) -> dict[str, Any]:
    out = copy_eeg(EEG)
    models = ensure_model_list(out, component_count(out))
    coordformat = str(dipfit_record(out).get("coordformat", "")).lower()
    axis = 0 if coordformat == "mni" else 1
    for component in components:
        model = models[component - 1]
        first = np.asarray(model.get("posxyz", [40.0, 0.0, 30.0]), dtype=float).reshape(-1, 3)[0]
        first[axis] = abs(first[axis]) or 40.0
        second = first.copy()
        second[axis] = -second[axis]
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


def _history_grid(value: Any, default: np.ndarray) -> Any:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return parse_grid_values(value, default=default).tolist()


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _source_points_from_option(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        if "pos" not in value:
            raise ValueError("sourcemodel dictionaries must contain a 'pos' field")
        return np.asarray(value["pos"], dtype=float)
    array = np.asarray(value if not isinstance(value, str) else [], dtype=float)
    if array.size:
        return array.reshape(-1, 3)
    if isinstance(value, str) and value.strip().lower() in {"loreta-key", "loreta_key"}:
        return loreta_key_source_points()
    if not isinstance(value, str):
        raise ValueError("sourcemodel must be source points, a {'pos': ...} dict, or a supported file")
    path = value.strip()
    if not path:
        raise ValueError(
            "pop_leadfield: sourcemodel is empty. "
            "Pass source points, a file, or sourcemodel='loreta-key' for EEGPrep's in-head grid."
        )
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.asarray(np.load(path), dtype=float).reshape(-1, 3)
    if lower.endswith(".npz"):
        with np.load(path) as data:
            key = "pos" if "pos" in data else next(iter(data.files))
            return np.asarray(data[key], dtype=float).reshape(-1, 3)
    if lower.endswith(".mat"):
        data = loadmat(path, squeeze_me=True, struct_as_record=False)
        points = _find_mat_points(data)
        if points is not None:
            return points
    if lower.endswith((".head", ".brik", ".nii", ".nii.gz")):
        raise DIPFITUnavailableError(
            "AFNI/NIfTI source-model leadfield computation requires atlas loading plus head-model clipping; "
            "use load_afni_atlas explicitly to inspect atlas points first."
        )
    raise DIPFITUnavailableError(f"Unsupported source model file for standalone pop_leadfield: {path}")


def _find_mat_points(data: dict[str, Any]) -> np.ndarray | None:
    for key in ("pos", "vertices", "Vertices"):
        if key in data:
            points = np.asarray(data[key], dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
    for value in data.values():
        if hasattr(value, "pos"):
            points = np.asarray(value.pos, dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
        if hasattr(value, "vertices"):
            points = np.asarray(value.vertices, dtype=float)
            if points.ndim == 2 and points.shape[1] >= 3:
                return points[:, :3]
    return None


__all__ = [
    "pop_dipfit_batch",
    "pop_dipfit_manual",
    "pop_dipfit_gridsearch",
    "pop_dipfit_gridsearch_dialog_spec",
    "pop_dipfit_headmodel",
    "pop_dipfit_headmodel_dialog_spec",
    "pop_dipfit_loreta",
    "pop_dipfit_loreta_dialog_spec",
    "pop_dipfit_nonlinear",
    "pop_dipfit_nonlinear_dialog_spec",
    "pop_leadfield",
    "pop_leadfield_dialog_spec",
    "pop_multifit",
    "pop_multifit_dialog_spec",
]

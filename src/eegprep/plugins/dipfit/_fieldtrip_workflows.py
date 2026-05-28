"""DIPFIT workflows that need FieldTrip before standalone execution."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.plugins.dipfit._utils import (
    component_count,
    ensure_channel_locations,
    ensure_dipfit_settings,
    ensure_ica,
    unavailable,
)


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
    """Validate coarse-fit inputs, then fail clearly until FieldTrip is ported."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    ensure_dipfit_settings(EEG)
    if gui is None:
        gui = select is None
    if gui:
        result = inputgui(pop_dipfit_gridsearch_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
    unavailable("pop_dipfit_gridsearch")


def pop_dipfit_nonlinear(
    EEG: dict[str, Any] | None = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Validate manual fine-fit prerequisites, then fail clearly until ported."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    ensure_dipfit_settings(EEG)
    if gui is None:
        gui = True
    if gui:
        result = inputgui(pop_dipfit_nonlinear_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
    unavailable("pop_dipfit_nonlinear")


def pop_multifit(
    EEG: dict[str, Any] | None = None,
    comps: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Validate automated multi-component fitting, then fail clearly until ported."""
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
    unavailable("pop_multifit")


def pop_leadfield(
    EEG: dict[str, Any] | None = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Validate leadfield settings, then fail clearly until FieldTrip is ported."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_dipfit_settings(EEG)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = not options
    if gui:
        result = inputgui(pop_leadfield_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(_leadfield_options_from_result(result))
    if not options.get("sourcemodel"):
        raise ValueError("sourcemodel is required to compute a leadfield matrix")
    unavailable("pop_leadfield")


def pop_dipfit_loreta(
    EEG: dict[str, Any] | None = None,
    select: Any = None,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Validate LORETA prerequisites, then fail clearly until FieldTrip is ported."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    dipfit = ensure_dipfit_settings(EEG)
    if not dipfit.get("sourcemodel"):
        raise ValueError("You need to compute a Leadfield matrix first")
    if str(dipfit.get("coordformat", "")).lower() != "mni":
        raise ValueError("For this function, you must use the template BEM model MNI in dipole fit settings")
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
        known_differences=("FieldTrip grid search is not bundled in standalone EEGPrep yet.",),
    )


def pop_dipfit_nonlinear_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return a compact version of EEGLAB's manual fine-fit dialog."""
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
            ControlSpec("edit", tag="component", value="1"),
            ControlSpec("pushbutton", "Plot map", tag="plotmap", enabled=False),
            ControlSpec("text", "Residual variance = "),
            ControlSpec("text", "not fit", tag="relvar"),
            ControlSpec("spacer"),
            ControlSpec("text", "dipole"),
            ControlSpec("text", "fit"),
            ControlSpec("text", "position"),
            ControlSpec("text", "moment"),
            ControlSpec("spacer"),
            ControlSpec("text", "1", tag="dip1"),
            ControlSpec("checkbox", tag="dip1sel", value=True),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="dip1pos", value="0 0 0"),
            ControlSpec("edit", tag="dip1mom", value="0 0 0"),
            ControlSpec("pushbutton", "Flip (in|out)", tag="flip1", enabled=False),
            ControlSpec("text", "#2", tag="dip2"),
            ControlSpec("checkbox", tag="dip2sel", value=False),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="dip2pos", value="0 0 0"),
            ControlSpec("edit", tag="dip2mom", value="0 0 0"),
            ControlSpec("pushbutton", "Flip (in|out)", tag="flip2", enabled=False),
            ControlSpec("checkbox", "Symmetry constrain for dipole #2", tag="dip2sym", value=True),
            ControlSpec("pushbutton", "Fit dipole(s)' position & moment", tag="fitposition", enabled=False),
            ControlSpec("pushbutton", "OR fit only dipole(s)' moment", tag="fitmoment", enabled=False),
            ControlSpec("pushbutton", "Plot dipole(s)", tag="plotdip", enabled=False),
        ),
        known_differences=("Fit/plot callbacks are disabled until the FieldTrip fitting backend is ported.",),
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
            ControlSpec("checkbox", "Plot resulting dipoles using dipplot (check)", tag="dipplot", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "plotting options", tag="plot_label", enabled=False),
            ControlSpec("edit", tag="plotopt", value="'normlen' 'on'", enabled=False),
            ControlSpec("pushbutton", "Help", tag="plot_help", enabled=False),
        ),
        known_differences=("Autofit requires FieldTrip grid search and nonlinear fitting, which are not bundled yet.",),
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
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "File name"),
            ControlSpec("edit", tag="sourcemodel", value=""),
            ControlSpec("pushbutton", "...", tag="source_browse", enabled=False),
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
            "Leadfield computation requires FieldTrip source-model preparation and is not bundled yet.",
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
    return {
        "sourcemodel": result.get("sourcemodel", ""),
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


__all__ = [
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

"""DIPFIT pop_dipfit_headmodel workflow and preserved GUI specification."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.plugins.dipfit._utils import unavailable


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
    unavailable(
        "pop_dipfit_headmodel",
        "MRI segmentation into a boundary-element head model is not implemented.",
        "Use pop_dipfit_settings with model='standardBESA' or model='standardBEM'.",
    )


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

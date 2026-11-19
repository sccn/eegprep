"""Unsupported source-analysis feature with its preserved GUI specification."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.plugins.dipfit._utils import unavailable


def pop_leadfield(EEG=None, *args, gui=None, renderer=None, return_com=False, **kwargs):
    """Report this unsupported source-analysis feature without changing the dataset."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = not args and not kwargs
    if gui and inputgui(pop_leadfield_dialog_spec(EEG), renderer=renderer) is None:
        return (EEG, "") if return_com else EEG
    unavailable("pop_leadfield")


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
            "Distributed source leadfields are unsupported. pop_leadfield raises NotImplementedError.",
            "Colin27, ICBM152, and AFNI atlas files are not bundled. MRI source-model construction and eLORETA are not implemented.",
        ),
    )

"""Unsupported source-analysis feature with its preserved GUI specification."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.dipfit._utils import unavailable


def pop_dipfit_loreta(EEG=None, *args, gui=None, renderer=None, return_com=False, **kwargs):
    """Report this unsupported source-analysis feature without changing the dataset."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = not args and not kwargs
    if gui and inputgui(pop_dipfit_loreta_dialog_spec(EEG), renderer=renderer) is None:
        return (EEG, "") if return_com else EEG
    unavailable("pop_dipfit_loreta")


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

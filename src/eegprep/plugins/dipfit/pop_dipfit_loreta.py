"""LORETA prerequisite checks for a stored DIPFIT leadfield."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.plugins.dipfit._utils import (
    ensure_channel_locations,
    ensure_dipfit_settings,
    ensure_ica,
    unavailable,
)


def pop_dipfit_loreta(EEG=None, select=None, *args, gui=None, renderer=None, return_com=False, **kwargs):
    """Check LORETA prerequisites, then report that eLORETA is not available."""
    if EEG is None:
        return (None, "") if return_com else None
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    dipfit = ensure_dipfit_settings(EEG)
    if not dipfit.get("sourcemodel"):
        raise ValueError(
            "pop_dipfit_loreta: no leadfield is stored in EEG.dipfit.sourcemodel. "
            "Run pop_leadfield first. The LORETA-KEY choice builds the in-head grid."
        )
    if str(dipfit.get("coordformat", "")).lower() != "mni":
        raise ValueError(
            "pop_dipfit_loreta: coordformat must be MNI. "
            "Run pop_dipfit_settings with model='standardBEM' before pop_dipfit_loreta."
        )
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is None:
        gui = select is None and not options
    if gui:
        result = inputgui(pop_dipfit_loreta_dialog_spec(EEG), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(_loreta_options_from_result(result))
    unavailable(
        "pop_dipfit_loreta",
        "eLORETA source analysis is not implemented.",
        "Use pop_multifit for an equivalent dipole or a symmetric pair on this dataset.",
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


def _loreta_options_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ft_sourceanalysis_params": result.get("ft_sourceanalysis_params", ""),
        "ft_sourceplot_params": result.get("ft_sourceplot_params", ""),
    }

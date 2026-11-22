"""DIPFIT pop_dipfit_gridsearch workflow and preserved GUI specification."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.dipfit._utils import (
    component_count,
    dipfit_history,
    ensure_channel_locations,
    ensure_dipfit_settings,
    ensure_ica,
    one_based_indices,
)

from eegprep.plugins.dipfit.dipfit_gridsearch import (
    default_search_grids,
    dipfit_gridsearch as run_gridsearch,
    parse_grid_values,
)


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
    """Run a coarse grid search for ICA components."""
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
        select = result.get("select", "")
        xgrid = result.get("xgrid", "")
        ygrid = result.get("ygrid", "")
        zgrid = result.get("zgrid", "")
        reject = result.get("reject", "")
    count = component_count(EEG)
    components = one_based_indices(select, limit=count, default_all=True)
    reject_value = _blank_to_none(reject)
    x_default, y_default, z_default = default_search_grids(EEG)
    out = run_gridsearch(EEG, component=components, xgrid=xgrid, ygrid=ygrid, zgrid=zgrid, reject=reject_value)
    command = dipfit_history(
        "pop_dipfit_gridsearch",
        components,
        _history_grid(xgrid, x_default),
        _history_grid(ygrid, y_default),
        _history_grid(zgrid, z_default),
        None if reject_value is None else float(reject_value),
    )
    return (out, command) if return_com else out


def pop_dipfit_gridsearch_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like coarse dipole fit dialog."""
    ncomp = max(component_count(EEG), 1)
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
            "standardBEM uses the packaged dipoli leadfield. standardBESA uses the four-shell BESA leadfield.",
        ),
    )


def _history_grid(value: Any, default: np.ndarray) -> Any:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return parse_grid_values(value, default=default).tolist()


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str) and not value.strip():
        return None
    return value

"""DIPFIT plugin menu spec for the EEGPrep main window.

TODO: replace placeholder actions with Python DIPFIT ports when available.
"""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item


def dipfit_menu() -> MenuItemSpec:
    """Return the EEGLAB DIPFIT Tools submenu."""
    return menu_item(
        "Source localization using DIPFIT",
        tag="dipfit",
        userdata="startup:off;study:on",
        separator=True,
        origin="dipfit",
        children=[
            menu_item("Head model and settings", action="pop_dipfit_settings", userdata="startup:off;study:on", origin="dipfit"),
            menu_item("Create a head model from an MRI", action="pop_dipfit_headmodel", userdata="startup:off;study:off", origin="dipfit"),
            menu_item("Component dipole coarse fit", action="pop_dipfit_gridsearch", userdata="startup:off", separator=True, origin="dipfit"),
            menu_item("Component dipole fine fit", action="pop_dipfit_nonlinear", userdata="startup:off", origin="dipfit"),
            menu_item("Component dipole plot ", action="pop_dipplot", userdata="startup:off", origin="dipfit"),
            menu_item("Component dipole autofit", action="pop_multifit", userdata="startup:off;study:on", origin="dipfit"),
            menu_item("Distributed source Leadfield matrix", action="pop_leadfield", userdata="startup:off;study:on", separator=True, origin="dipfit"),
            menu_item("Distributed source component modelling", action="pop_dipfit_loreta", userdata="startup:off", origin="dipfit"),
        ],
    )

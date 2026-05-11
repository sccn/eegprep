"""ICLabel and viewprops plugin menu specs for the EEGPrep main window."""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item


def iclabel_menu() -> MenuItemSpec:
    """Return the EEGLAB ICLabel Tools submenu."""
    return menu_item(
        "Classify components using ICLabel",
        userdata="startup:off;study:on;roi:off",
        origin="ICLabel",
        children=[
            menu_item("Label components", action="pop_iclabel", userdata="startup:off;study:on", origin="ICLabel"),
            menu_item("Flag components as artifacts", action="pop_icflag", userdata="startup:off;study:on", origin="ICLabel"),
            menu_item("View extended component properties", action="pop_viewprops:components", origin="ICLabel"),
        ],
    )


def viewprops_plot_menus() -> tuple[MenuItemSpec, MenuItemSpec]:
    """Return viewprops Plot menu additions."""
    return (
        menu_item("View extended channel properties", action="pop_viewprops:channels", origin="viewprops"),
        menu_item("View extended component properties", action="pop_viewprops:components", origin="viewprops"),
    )

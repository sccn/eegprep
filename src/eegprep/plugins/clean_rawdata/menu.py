"""clean_rawdata plugin menu spec for the EEGPrep main window."""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item


def clean_rawdata_menu() -> MenuItemSpec:
    """Return the EEGLAB clean_rawdata Tools menu item."""
    return menu_item(
        "Reject data using Clean Rawdata and ASR",
        action="pop_clean_rawdata",
        userdata="startup:off;epoch:off;study:on",
        origin="clean_rawdata",
    )

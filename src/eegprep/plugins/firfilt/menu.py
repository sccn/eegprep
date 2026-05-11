"""firfilt plugin menu specs for the EEGPrep main window."""

from __future__ import annotations

from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_item


def firfilt_filter_items() -> tuple[MenuItemSpec, ...]:
    """Return firfilt items inserted under Tools > Filter the data."""
    return (
        menu_item("Basic FIR filter (new, default)", action="pop_eegfiltnew", userdata="study:on", origin="firfilt"),
        menu_item("Windowed sinc FIR filter", action="pop_firws", origin="firfilt"),
        menu_item("Parks-McClellan (equiripple) FIR filter", action="pop_firpm", origin="firfilt"),
        menu_item("Moving average FIR filter", action="pop_firma", origin="firfilt"),
    )

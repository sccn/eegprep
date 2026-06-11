"""EEGLAB-style pop_kaiserbeta wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._pop_common import numeric_or_none, value_history_command
from eegprep.plugins.firfilt.kaiserbeta import kaiserbeta


def pop_kaiserbeta(
    dev: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_dev: bool = False,
    return_com: bool = False,
):
    """Estimate Kaiser window beta from passband ripple."""
    if gui is None:
        gui = dev is None
    if gui:
        result = inputgui(pop_kaiserbeta_dialog_spec(), renderer=renderer)
        if result is None:
            empty = (None, None) if return_dev else None
            return (empty, "") if return_com else empty
        dev = result.get("dev")
    parsed = numeric_or_none(dev)
    if parsed is None:
        raise ValueError("Not enough input arguments.")
    beta = kaiserbeta(parsed)
    output = (beta, parsed) if return_dev else beta
    command = value_history_command("pop_kaiserbeta", [parsed], assignment="beta")
    return (output, command) if return_com else output


def pop_kaiserbeta_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_kaiserbeta``."""
    return DialogSpec(
        title="Estimate Kaiser window beta -- pop_kaiserbeta()",
        function_name="pop_kaiserbeta",
        eeglab_source="plugins/firfilt/pop_kaiserbeta.m",
        geometry=((1, 1),),
        geomvert=(1,),
        size=(420, 144),
        help_text="pophelp('pop_kaiserbeta')",
        controls=(
            ControlSpec("text", "Max passband deviation/ripple:"),
            ControlSpec("edit", tag="dev", value=""),
        ),
    )

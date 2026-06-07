"""EEGLAB-style pop_firwsord wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import WINDOW_TYPES
from eegprep.plugins.firfilt._pop_common import numeric_or_none, value_history_command
from eegprep.plugins.firfilt.firwsord import firwsord


def pop_firwsord(
    wtype: Any = None,
    fs: Any = None,
    df: Any = None,
    dev: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_dev: bool = False,
    return_com: bool = False,
):
    """Estimate a windowed-sinc FIR order."""
    if gui is None:
        gui = df is None
    wtypes = list(WINDOW_TYPES)
    if gui:
        result = inputgui(pop_firwsord_dialog_spec(), renderer=renderer)
        if result is None:
            empty = (None, None) if return_dev else None
            return (empty, "") if return_com else empty
        fs = result.get("fs")
        wtype = wtypes[int(result.get("wtype", 3)) - 1]
        df = result.get("df")
        dev = result.get("dev")
    if wtype is None or str(wtype).strip() == "":
        wtype = "hamming"
    if fs is None or str(fs).strip() == "":
        fs = 2
    parsed_df = numeric_or_none(df)
    if parsed_df is None:
        raise ValueError("Not enough input arguments.")
    parsed_fs = float(fs)
    parsed_wtype = str(wtype).lower()
    parsed_dev = numeric_or_none(dev)
    order, out_dev = firwsord(parsed_wtype, parsed_fs, parsed_df, parsed_dev)
    output = (int(order), out_dev) if return_dev else int(order)
    values = [parsed_wtype, parsed_fs, parsed_df] + ([] if parsed_dev is None else [parsed_dev])
    command = value_history_command("pop_firwsord", values, assignment="m")
    return (output, command) if return_com else output


def pop_firwsord_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_firwsord``."""
    return DialogSpec(
        title="Estimate filter order -- pop_firwsord()",
        function_name="pop_firwsord",
        eeglab_source="plugins/firfilt/pop_firwsord.m",
        geometry=((1, 1), (1, 1), (1, 1), (1, 1)),
        geomvert=(1, 1, 1, 1),
        size=(480, 232),
        help_text="pophelp('pop_firwsord')",
        controls=(
            ControlSpec("text", "Sampling frequency:"),
            ControlSpec("edit", tag="fs", value="2"),
            ControlSpec("text", "Window type:"),
            ControlSpec(
                "popupmenu",
                "|".join(["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"]),
                tag="wtype",
                value=3,
                callback=CallbackSpec(
                    "toggle_index_enabled",
                    params={"source": "wtype", "enabled_index": 5, "targets": ("dev", "dev_label")},
                ),
            ),
            ControlSpec("text", "Transition bandwidth (Hz):"),
            ControlSpec("edit", tag="df", value=""),
            ControlSpec("text", "Max passband deviation/ripple:", tag="dev_label", enabled=False),
            ControlSpec("edit", tag="dev", value="", enabled=False),
        ),
    )

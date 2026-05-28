"""EEGLAB-style pop_firpm wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import FILTER_TYPES, apply_fir_filter, design_firpm
from eegprep.plugins.firfilt._pop_common import (
    history_command,
    int_or_none,
    normalize_pop_options,
    numeric_or_none,
    vector_or_none,
)


def pop_firpm(
    EEG: Any,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Filter EEG data using a Parks-McClellan equiripple FIR filter."""
    if EEG is None:
        raise ValueError("Cannot process empty dataset")
    options = normalize_pop_options(args, kwargs)
    if gui is None:
        gui = not options
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(result)
    parsed = _parsed_options(options)
    if isinstance(EEG, list):
        output = [pop_firpm(item, gui=False, **parsed) for item in EEG]
        command = history_command("pop_firpm", parsed)
        return (output, command) if return_com else output
    b = design_firpm(float(EEG["srate"]), **parsed)
    output = apply_fir_filter(EEG, b)
    command = history_command("pop_firpm", parsed)
    return (output, command) if return_com else output


def pop_firpm_dialog_spec(_EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_firpm``."""
    return DialogSpec(
        title="Filter the data -- pop_firpm()",
        function_name="pop_firpm",
        eeglab_source="plugins/firfilt/pop_firpm.m",
        geometry=(
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
        ),
        geomvert=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        size=(962, 440),
        help_text="pophelp('pop_firpm')",
        controls=(
            ControlSpec("text", "Cutoff frequency(ies) [hp lp] (~-6 dB; Hz):"),
            ControlSpec("edit", tag="fcutoff", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Transition band width:"),
            ControlSpec("edit", tag="ftrans", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Filter type:"),
            ControlSpec("popupmenu", "|".join(FILTER_TYPES), tag="ftype", value=1),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Passband weight:"),
            ControlSpec("edit", tag="wtpass", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Stopband weight:"),
            ControlSpec("edit", tag="wtstop", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Filter order (mandatory even):"),
            ControlSpec("edit", tag="forder", value=""),
            ControlSpec("pushbutton", "Estimate", tag="orderpush"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Plot filter responses", tag="plotpush"),
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_firpm_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {"ftype": FILTER_TYPES[int(result.get("ftype", 1)) - 1]}
    for key in ("fcutoff",):
        value = vector_or_none(result.get(key))
        if value is not None:
            options[key] = value
    for key in ("ftrans", "wtpass", "wtstop"):
        value = numeric_or_none(result.get(key))
        if value is not None:
            options[key] = value
    forder = int_or_none(result.get("forder"))
    if forder is not None:
        options["forder"] = forder
    return options


def _parsed_options(options: dict[str, Any]) -> dict[str, Any]:
    fcutoff = vector_or_none(options.get("fcutoff"))
    ftrans = numeric_or_none(options.get("ftrans"))
    forder = int_or_none(options.get("forder"))
    if fcutoff is None or ftrans is None or forder is None or not options.get("ftype"):
        raise ValueError("Not enough input arguments.")
    parsed: dict[str, Any] = {
        "fcutoff": fcutoff,
        "ftrans": ftrans,
        "ftype": str(options["ftype"]).lower(),
        "forder": forder,
    }
    for key in ("wtpass", "wtstop"):
        value = numeric_or_none(options.get(key))
        if value is not None:
            parsed[key] = value
    return parsed

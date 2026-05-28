"""EEGLAB-style pop_firws wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import FILTER_TYPES, WINDOW_TYPES, apply_fir_filter, design_firws
from eegprep.plugins.firfilt._pop_common import (
    bool_value,
    has_value,
    history_command,
    int_or_none,
    normalize_pop_options,
    numeric_or_none,
    vector_or_none,
)


def pop_firws(
    EEG: Any,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Filter EEG data using a windowed-sinc FIR filter."""
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
        output = [pop_firws(item, gui=False, **parsed) for item in EEG]
        command = history_command("pop_firws", parsed)
        return (output, command) if return_com else output
    design_options = {
        key: value for key, value in parsed.items() if key not in {"channels", "chantype", "plotfresp", "usefftfilt"}
    }
    b = design_firws(float(EEG["srate"]), **design_options)
    output = apply_fir_filter(
        EEG,
        b,
        channels=parsed.get("channels"),
        chantype=parsed.get("chantype"),
        causal=bool_value(parsed.get("minphase")),
        usefftfilt=bool_value(parsed.get("usefftfilt")),
    )
    command = history_command("pop_firws", parsed)
    return (output, command) if return_com else output


def pop_firws_dialog_spec(_EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_firws``."""
    return DialogSpec(
        title="Filter the data -- pop_firws()",
        function_name="pop_firws",
        eeglab_source="plugins/firfilt/pop_firws.m",
        geometry=(
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1, 1.5),
            (1, 1.5),
            (1,),
            (1, 0.75, 0.75),
        ),
        geomvert=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        size=(1395, 476),
        help_text="pophelp('pop_firws')",
        controls=(
            ControlSpec("text", "Cutoff frequency(ies) [hp lp] (-6 dB; Hz):"),
            ControlSpec("edit", tag="fcutoff", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Filter type:"),
            ControlSpec("popupmenu", "|".join(["Bandpass", "Highpass", "Lowpass", "Bandstop"]), tag="ftype", value=1),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Window type:"),
            ControlSpec(
                "popupmenu", "|".join(["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"]), tag="wtype", value=3
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Kaiser window beta:", tag="warg_label", enabled=False),
            ControlSpec("edit", tag="warg", value="", enabled=False),
            ControlSpec(
                "pushbutton",
                "Estimate",
                tag="wargpush",
                enabled=False,
                tooltip="Kaiser beta estimation is not yet implemented in EEGPrep.",
            ),
            ControlSpec("text", "Filter order (mandatory even):"),
            ControlSpec("edit", tag="forder", value=""),
            ControlSpec(
                "pushbutton",
                "Estimate",
                enabled=False,
                tooltip="Filter-order estimation is not yet implemented in EEGPrep.",
            ),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox", "Use minimum-phase converted causal filter (non-linear!)", tag="minphase", value=False
            ),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Use frequency domain filtering (faster for high filter orders > ~2000)",
                tag="usefftfilt",
                value=False,
            ),
            ControlSpec("edit", tag="dev", value="", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec(
                "pushbutton",
                "Plot filter responses",
                enabled=False,
                tooltip="Filter-response plotting from this dialog is not yet implemented in EEGPrep.",
            ),
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_firws_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {
        "ftype": FILTER_TYPES[int(result.get("ftype", 1)) - 1],
        "wtype": WINDOW_TYPES[int(result.get("wtype", 3)) - 1],
    }
    cutoff = vector_or_none(result.get("fcutoff"))
    if cutoff is not None:
        options["fcutoff"] = cutoff
    forder = int_or_none(result.get("forder"))
    if forder is not None:
        options["forder"] = forder
    warg = numeric_or_none(result.get("warg"))
    if warg is not None:
        options["warg"] = warg
    for key in ("minphase", "usefftfilt"):
        if bool_value(result.get(key)):
            options[key] = True
    return options


def _parsed_options(options: dict[str, Any]) -> dict[str, Any]:
    fcutoff = vector_or_none(options.get("fcutoff"))
    forder = int_or_none(options.get("forder"))
    if fcutoff is None or forder is None:
        raise ValueError("Not enough input arguments.")
    parsed: dict[str, Any] = {
        "fcutoff": fcutoff,
        "forder": forder,
        "ftype": str(options.get("ftype", "lowpass" if len(fcutoff) == 1 else "bandpass")).lower(),
        "wtype": str(options.get("wtype", "hamming")).lower(),
    }
    warg = numeric_or_none(options.get("warg"))
    if warg is not None:
        parsed["warg"] = warg
    for key in ("minphase", "usefftfilt", "plotfresp"):
        if bool_value(options.get(key)):
            parsed[key] = True
    for key in ("channels", "chantype"):
        if has_value(options.get(key)):
            parsed[key] = options[key]
    return parsed

"""Design and export xfir-compatible windowed-sinc FIR filters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import FILTER_TYPES, WINDOW_TYPES, design_firws
from eegprep.plugins.firfilt._pop_common import int_or_none, numeric_or_none, value_history_command, vector_or_none


def pop_xfirws(
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Design a FIRFilt windowed-sinc FIR and optionally export an ``.fir`` file."""
    options = _parse_options(args, kwargs)
    if gui is None:
        gui = not options
    if gui:
        result = inputgui(pop_xfirws_dialog_spec(), renderer=renderer)
        if result is None:
            empty = (np.array([]), 1)
            return (empty, "") if return_com else empty
        options.update(_options_from_gui(result))
    parsed = _parsed_options(options)
    b = design_firws(
        parsed["srate"],
        fcutoff=parsed["fcutoff"],
        forder=parsed["forder"],
        ftype=parsed["ftype"],
        wtype=parsed["wtype"],
        warg=parsed.get("warg"),
    )
    if parsed.get("filename"):
        _write_xfirws_file(b, parsed)
    values = []
    for key in ("srate", "fcutoff", "ftype", "wtype", "warg", "forder", "filename", "pathname"):
        if key in parsed and parsed[key] is not None:
            values.extend([key, parsed[key]])
    command = value_history_command("pop_xfirws", values, assignment="[b, a]")
    output = (b, 1)
    return (output, command) if return_com else output


def pop_xfirws_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_xfirws``."""
    return DialogSpec(
        title="Filter the data -- pop_firws()",
        function_name="pop_xfirws",
        eeglab_source="plugins/firfilt/pop_xfirws.m",
        geometry=(
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1, 0.75, 0.75),
            (1,),
            (1, 0.75, 0.75),
        ),
        geomvert=(1, 0.4, 1, 1, 0.4, 1, 1, 1, 0.4, 1),
        size=(980, 506),
        help_text="pophelp('pop_firws')",
        controls=(
            ControlSpec("text", "Sampling frequency (Hz):"),
            ControlSpec("edit", tag="srate", value="2"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Cutoff frequency(ies) [hp lp] (-6 dB; Hz):"),
            ControlSpec("edit", tag="fcutoff", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Filter type:"),
            ControlSpec("popupmenu", "|".join(["Bandpass", "Highpass", "Lowpass", "Bandstop"]), tag="ftype", value=1),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Window type:"),
            ControlSpec(
                "popupmenu",
                "|".join(["Rectangular", "Hann", "Hamming", "Blackman", "Kaiser"]),
                tag="wtype",
                value=3,
                callback=CallbackSpec(
                    "toggle_index_enabled",
                    params={"source": "wtype", "enabled_index": 5, "targets": ("warg", "warg_label", "wargpush")},
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Kaiser window beta:", tag="warg_label", enabled=False),
            ControlSpec("edit", tag="warg", value="", enabled=False),
            ControlSpec(
                "pushbutton",
                "Estimate",
                tag="wargpush",
                enabled=False,
                callback=CallbackSpec("fir_kaiser_beta", params={"button": "wargpush", "target": "warg"}),
            ),
            ControlSpec("text", "Filter order (mandatory even):"),
            ControlSpec("edit", tag="forder", value=""),
            ControlSpec(
                "pushbutton",
                "Estimate",
                tag="forderpush",
                callback=CallbackSpec(
                    "firws_order",
                    params={
                        "button": "forderpush",
                        "target": "forder",
                        "srate": "srate",
                        "wtype": "wtype",
                        "dev": "dev",
                    },
                ),
            ),
            ControlSpec("edit", tag="dev", value="", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec(
                "pushbutton",
                "Plot filter responses",
                tag="plotpush",
                callback=CallbackSpec(
                    "fir_response_plot", params={"button": "plotpush", "design": "firws", "srate": "srate"}
                ),
            ),
        ),
    )


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    options = {str(key).lower(): value for key, value in kwargs.items()}
    if args:
        if len(args) % 2:
            raise ValueError("Key/value arguments must be in pairs")
        for index in range(0, len(args), 2):
            options[str(args[index]).lower()] = args[index + 1]
    return options


def _options_from_gui(result: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "srate": numeric_or_none(result.get("srate")),
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
    return options


def _parsed_options(options: dict[str, Any]) -> dict[str, Any]:
    srate = numeric_or_none(options.get("srate")) or 2.0
    fcutoff = vector_or_none(options.get("fcutoff"))
    forder = int_or_none(options.get("forder"))
    if fcutoff is None or forder is None:
        raise ValueError("Not enough input arguments.")
    parsed: dict[str, Any] = {
        "srate": float(srate),
        "fcutoff": fcutoff,
        "forder": forder,
        "ftype": str(options.get("ftype", "lowpass" if len(fcutoff) == 1 else "bandpass")).lower(),
        "wtype": str(options.get("wtype", "hamming")).lower(),
    }
    warg = numeric_or_none(options.get("warg"))
    if warg is not None:
        parsed["warg"] = warg
    for key in ("filename", "pathname"):
        if options.get(key):
            parsed[key] = Path(options[key]) if key == "pathname" else str(options[key])
    return parsed


def _write_xfirws_file(b: np.ndarray, options: dict[str, Any]) -> None:
    pathname = Path(options.get("pathname", "."))
    path = pathname / str(options["filename"])
    lines = [
        "[author]",
        "pop_xfirws 2.0",
        "",
        "[fir design]",
        "method  fourier",
        f"type    {options['ftype']}",
        f"fsample {options['srate']:.6f}",
        f"length  {options['forder'] + 1}",
    ]
    lines.extend(f"fcrit{index}  {value:.6f}" for index, value in enumerate(options["fcutoff"], start=1))
    warg = "" if options.get("warg") is None else str(options["warg"])
    lines.extend(["window  " + options["wtype"] + " " + warg, "", "[fir]", str(options["forder"] + 1)])
    lines.extend(f"{value: 18.10e}" for value in b)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

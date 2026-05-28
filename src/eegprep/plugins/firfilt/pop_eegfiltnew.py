"""EEGLAB-style pop_eegfiltnew wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import apply_fir_filter, design_eegfiltnew
from eegprep.plugins.firfilt._pop_common import (
    bool_value,
    channel_controls,
    history_command,
    int_or_none,
    normalize_pop_options,
    numeric_or_none,
    vector_or_none,
)


def pop_eegfiltnew(
    EEG: Any,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Filter EEG data using EEGLAB's Hamming-windowed FIR defaults."""
    if EEG is None:
        raise ValueError("Cannot filter empty dataset.")
    options = normalize_pop_options(
        args,
        kwargs,
        positional=("locutoff", "hicutoff", "filtorder", "revfilt", "usefft", "plotfreqz", "minphase", "usefftfilt"),
    )
    if gui is None:
        gui = not options
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options.update(result)

    locutoff = numeric_or_none(options.get("locutoff"))
    hicutoff = numeric_or_none(options.get("hicutoff"))
    filtorder = int_or_none(options.get("filtorder"))
    revfilt = bool_value(options.get("revfilt"))
    usefft = bool_value(options.get("usefft"))
    usefftfilt = bool_value(options.get("usefftfilt"))
    minphase = bool_value(options.get("minphase"))
    plotfreqz = bool_value(options.get("plotfreqz"))
    channels = options.get("channels")
    chantype = options.get("chantype")
    if usefft:
        raise ValueError("FFT filtering is not supported; use usefftfilt for FIR frequency-domain filtering.")

    command_options = _command_options(
        locutoff=locutoff,
        hicutoff=hicutoff,
        filtorder=filtorder,
        revfilt=revfilt,
        plotfreqz=plotfreqz,
        minphase=minphase,
        usefftfilt=usefftfilt,
        channels=channels,
        chantype=chantype,
    )
    if isinstance(EEG, list):
        output = [pop_eegfiltnew(item, gui=False, **command_options) for item in EEG]
        command = history_command("pop_eegfiltnew", command_options)
        return (output, command) if return_com else output

    b, _metadata = design_eegfiltnew(
        float(EEG["srate"]),
        locutoff=locutoff,
        hicutoff=hicutoff,
        filtorder=filtorder,
        revfilt=revfilt,
        minphase=minphase,
    )
    output = apply_fir_filter(EEG, b, channels=channels, chantype=chantype, causal=minphase, usefftfilt=usefftfilt)
    command = history_command("pop_eegfiltnew", command_options)
    return (output, command) if return_com else output


def pop_eegfiltnew_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_eegfiltnew``."""
    controls = (
        ControlSpec("text", "Lower edge of the frequency pass band (Hz)"),
        ControlSpec("edit", tag="locutoff", value=""),
        ControlSpec("text", "Higher edge of the frequency pass band (Hz)"),
        ControlSpec("edit", tag="hicutoff", value=""),
        ControlSpec("text", "FIR Filter order (Mandatory even. Default is automatic*)"),
        ControlSpec("edit", tag="filtorder", value=""),
        ControlSpec(
            "text",
            "*See help text for a description of the default filter order heuristic.\nManual definition is recommended.",
        ),
        ControlSpec("checkbox", "Notch filter the data instead of pass band", tag="revfilt", value=False),
        ControlSpec(
            "checkbox",
            "Use minimum-phase converted causal filter (non-linear!; beta)",
            tag="minphase",
            value=False,
        ),
        ControlSpec("checkbox", "Plot frequency response", tag="plotfreqz", value=True),
        *channel_controls(EEG),
        ControlSpec(
            "checkbox",
            "Use frequency domain filtering (faster for high filter orders > ~2000)",
            tag="usefftfilt",
            value=False,
        ),
    )
    return DialogSpec(
        title="Filter the data -- pop_eegfiltnew()",
        function_name="pop_eegfiltnew",
        eeglab_source="plugins/firfilt/pop_eegfiltnew.m",
        geometry=((3, 1), (3, 1), (3, 1), (1,), (1,), (1,), (1,), (2, 1.5, 0.5), (2, 1.5, 0.5), (1,)),
        geomvert=(1, 1, 1, 2, 1, 1, 1, 1, 1, 1),
        size=(654, 511),
        help_text="pophelp('pop_eegfiltnew')",
        controls=controls,
        row_spacing=4,
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_eegfiltnew_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {}
    for key in ("locutoff", "hicutoff"):
        value = numeric_or_none(result.get(key))
        if value is not None:
            options[key] = value
    filtorder = int_or_none(result.get("filtorder"))
    if filtorder is not None:
        options["filtorder"] = filtorder
    for key in ("revfilt", "minphase", "plotfreqz", "usefftfilt"):
        if bool_value(result.get(key)):
            options[key] = True
    chantype = str(result.get("chantype", "")).strip()
    channels = str(result.get("channels", "")).strip()
    if chantype:
        options["chantype"] = chantype.split()
    elif channels:
        parsed = vector_or_none(channels)
        options["channels"] = parsed if parsed is not None else channels.split()
    return options


def _command_options(**values: Any) -> dict[str, Any]:
    ordered = {}
    for key in (
        "locutoff",
        "hicutoff",
        "filtorder",
        "revfilt",
        "plotfreqz",
        "minphase",
        "usefftfilt",
        "channels",
        "chantype",
    ):
        value = values[key]
        if value is not None:
            ordered[key] = value
    return ordered

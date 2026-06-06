"""EEGLAB-style legacy pop_eegfilt wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.plugins.firfilt._filtering import apply_eegfilt_legacy, design_eegfilt_legacy
from eegprep.plugins.firfilt._pop_common import bool_value, int_or_none, numeric_or_none


def pop_eegfilt(
    EEG: Any,
    locutoff: Any = None,
    hicutoff: Any = None,
    filtorder: Any = None,
    revfilt: Any = 0,
    usefft: Any = 0,
    plotfreqz: Any = 0,
    firtype: str = "firls",
    causal: Any = 0,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Filter EEG data through the legacy ``pop_eegfilt`` interface."""
    if EEG is None:
        raise ValueError("Pop_eegfilt() error: cannot filter an empty dataset")
    if gui is None:
        gui = locutoff is None and hicutoff is None
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        locutoff = result["locutoff"]
        hicutoff = result["hicutoff"]
        filtorder = result.get("filtorder")
        revfilt = result["revfilt"]
        usefft = result["usefft"]
        plotfreqz = result["plotfreqz"]
        firtype = result["firtype"]
        causal = result["causal"]

    parsed_locutoff = numeric_or_none(locutoff)
    parsed_hicutoff = numeric_or_none(hicutoff)
    locutoff = 0.0 if parsed_locutoff is None else float(parsed_locutoff)
    hicutoff = 0.0 if parsed_hicutoff is None else float(parsed_hicutoff)
    filtorder = int_or_none(filtorder)
    revfilt = bool_value(revfilt)
    usefft = bool_value(usefft)
    plotfreqz = bool_value(plotfreqz)
    causal = bool_value(causal)
    if locutoff == 0 and hicutoff == 0:
        return (EEG, "") if return_com else EEG
    if usefft:
        raise ValueError(
            "Legacy pop_eegfilt FFT filtering is not a standalone EEGPrep path. "
            "Use pop_eegfiltnew(..., usefftfilt=True) for frequency-domain FIR filtering."
        )

    if isinstance(EEG, list):
        output = [
            pop_eegfilt(item, locutoff, hicutoff, filtorder, revfilt, usefft, plotfreqz, firtype, causal)
            for item in EEG
        ]
        command = _history_command(locutoff, hicutoff, filtorder, revfilt, usefft, plotfreqz, firtype, causal)
        return (output, command) if return_com else output

    b, order = design_eegfilt_legacy(
        float(EEG["srate"]),
        locutoff=locutoff,
        hicutoff=hicutoff,
        filtorder=filtorder,
        revfilt=revfilt,
        firtype=firtype,
    )
    output = apply_eegfilt_legacy(EEG, b, causal=causal, filtorder=order)
    command = _history_command(locutoff, hicutoff, filtorder, revfilt, usefft, plotfreqz, firtype, causal)
    return (output, command) if return_com else output


def pop_eegfilt_dialog_spec(_EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for legacy ``pop_eegfilt``."""
    return DialogSpec(
        title="Filter the data -- pop_eegfilt()",
        function_name="pop_eegfilt",
        eeglab_source="functions/popfunc/pop_eegfilt.m",
        geometry=((3, 1), (3, 1), (3, 1), (1,), (1,), (1,), (1,), (1,), (1,)),
        geomvert=(1, 1, 1, 1, 1, 1, 1, 1, 1),
        size=(629, 440),
        help_text="pophelp('pop_eegfilt')",
        controls=(
            ControlSpec("text", "Lower edge of the frequency pass band (Hz)"),
            ControlSpec("edit", tag="locutoff", value=""),
            ControlSpec("text", "Higher edge of the frequency pass band (Hz)"),
            ControlSpec("edit", tag="hicutoff", value=""),
            ControlSpec("text", "FIR Filter order (default is automatic)"),
            ControlSpec("edit", tag="filtorder", value=""),
            ControlSpec("checkbox", "Notch filter the data instead of pass band", tag="revfilt", value=False),
            ControlSpec(
                "checkbox", "Use (sharper) FFT linear filter instead of FIR filtering", tag="usefft", value=False
            ),
            ControlSpec("text", "(Use the option above if you do not have the Signal Processing Toolbox)"),
            ControlSpec(
                "checkbox", "Use causal filter (useful when performing causal analysis)", tag="causal", value=False
            ),
            ControlSpec("checkbox", "Plot the filter frequency response", tag="plotfreqz", value=False),
            ControlSpec(
                "checkbox", "Use fir1 (check, recommended) or firls (uncheck, legacy)", tag="firtype", value=True
            ),
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_eegfilt_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    locutoff = numeric_or_none(result.get("locutoff")) or 0
    hicutoff = numeric_or_none(result.get("hicutoff")) or 0
    filtorder = int_or_none(result.get("filtorder"))
    revfilt = bool_value(result.get("revfilt"))
    if revfilt and (locutoff == 0 or hicutoff == 0):
        raise ValueError("Need both lower and higher edge for notch filter")
    return {
        "locutoff": locutoff,
        "hicutoff": hicutoff,
        "filtorder": filtorder,
        "revfilt": revfilt,
        "usefft": bool_value(result.get("usefft")),
        "causal": bool_value(result.get("causal")),
        "plotfreqz": bool_value(result.get("plotfreqz")),
        "firtype": "fir1" if bool_value(result.get("firtype"), default=True) else "firls",
    }


def _history_command(
    locutoff: float,
    hicutoff: float,
    filtorder: int | None,
    revfilt: bool,
    usefft: bool,
    plotfreqz: bool,
    firtype: str,
    causal: bool,
) -> str:
    return (
        "EEG = pop_eegfilt( EEG, "
        f"{format_history_value(locutoff)}, {format_history_value(hicutoff)}, "
        f"{format_history_value([] if filtorder is None else [filtorder], cell_for_sequence=None)}, "
        f"{format_history_value([int(revfilt)], cell_for_sequence=None)}, "
        f"{int(usefft)}, {int(plotfreqz)}, {format_history_value(firtype)}, {int(causal)});"
    )

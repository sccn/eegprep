"""EEGLAB-style pop_firma wrapper."""

from __future__ import annotations

from typing import Any

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.firfilt._filtering import apply_fir_filter, design_firma
from eegprep.plugins.firfilt._pop_common import history_command, int_or_none, normalize_pop_options


def pop_firma(
    EEG: Any,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Filter EEG data using a moving-average FIR filter."""
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
    forder = int_or_none(options.get("forder"))
    if forder is None:
        raise ValueError("Not enough input arguments")
    parsed = {"forder": forder}
    if isinstance(EEG, list):
        output = [pop_firma(item, gui=False, **parsed) for item in EEG]
        command = history_command("pop_firma", parsed)
        return (output, command) if return_com else output
    output = apply_fir_filter(EEG, design_firma(forder=forder))
    command = history_command("pop_firma", parsed)
    return (output, command) if return_com else output


def pop_firma_dialog_spec(_EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_firma``."""
    return DialogSpec(
        title="Filter the data -- pop_firma()",
        function_name="pop_firma",
        eeglab_source="plugins/firfilt/pop_firma.m",
        geometry=((1, 1, 1), (1,), (1, 1, 1)),
        geomvert=(1, 1, 1),
        size=(862, 231),
        help_text="pophelp('pop_firma')",
        controls=(
            ControlSpec("text", "Filter order (mandatory even):"),
            ControlSpec("edit", tag="forder", value=""),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
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
    result = inputgui(pop_firma_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    forder = int_or_none(result.get("forder"))
    if forder is None:
        raise ValueError("Not enough input arguments")
    return {"forder": forder}

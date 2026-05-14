"""EEGLAB-style pop wrapper for ICLabel."""

from __future__ import annotations

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.ICLabel.iclabel import iclabel


_VERSIONS = ("default", "lite", "beta")


def pop_iclabel(
    EEG,
    icversion: str | None = None,
    *,
    gui: bool | None = None,
    renderer=None,
    engine=None,
    return_com: bool = False,
):
    """Classify independent components using ICLabel."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = icversion is None
    if gui:
        result = _run_gui(renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        icversion = result["icversion"]
    icversion = "default" if icversion is None else str(icversion).lower()
    if icversion not in _VERSIONS:
        raise ValueError("icversion must be one of 'default', 'lite', or 'beta'")
    if isinstance(EEG, list):
        output = [pop_iclabel(item, icversion, gui=False, engine=engine) for item in EEG]
        command = _history_command(icversion)
        return (output, command) if return_com else output
    _require_ica(EEG)
    output = iclabel(EEG, algorithm=icversion, engine=engine)
    command = _history_command(icversion)
    return (output, command) if return_com else output


def pop_iclabel_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_iclabel``."""
    return DialogSpec(
        title="ICLabel",
        function_name="pop_iclabel",
        eeglab_source="plugins/ICLabel/pop_iclabel.m",
        geometry=((1,), (1,)),
        size=(356, 199),
        help_text="pophelp('pop_iclabel')",
        controls=(
            ControlSpec("text", "Select which icversion of ICLabel to use:"),
            ControlSpec("popupmenu", "Default (recommended)|Lite|Beta", tag="icversion", value=1),
        ),
    )


def _run_gui(renderer=None):
    result = inputgui(pop_iclabel_dialog_spec(), renderer=renderer)
    if result is None:
        return None
    index = int(result.get("icversion", 1)) - 1
    index = max(0, min(index, len(_VERSIONS) - 1))
    return {"icversion": _VERSIONS[index]}


def _require_ica(EEG):
    weights = EEG.get("icaweights")
    if weights is None or getattr(weights, "size", len(weights) if hasattr(weights, "__len__") else 0) == 0:
        raise ValueError("ICLabel requires an ICA decomposition. Run pop_runica first.")


def _history_command(icversion):
    return f"EEG = pop_iclabel(EEG, '{icversion}');"

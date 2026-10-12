"""EEGLAB-style pop wrapper for ICLabel."""

from __future__ import annotations

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.plugins.ICLabel.iclabel import iclabel, iclabel_async, require_sync_backend


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
    require_sync_backend(engine)
    icversion = _requested_icversion(EEG, icversion, gui, renderer)
    if icversion is None:
        return (EEG, "") if return_com else EEG
    output = [iclabel(dataset, algorithm=icversion, engine=engine) for dataset in _datasets_with_ica(EEG)]
    output = output if isinstance(EEG, list) else output[0]
    return (output, _history_command(icversion)) if return_com else output


async def pop_iclabel_async(
    EEG,
    icversion: str | None = None,
    *,
    gui: bool | None = None,
    renderer=None,
    engine=None,
    return_com: bool = False,
):
    """Classify independent components through an awaitable ICLabel backend."""
    icversion = _requested_icversion(EEG, icversion, gui, renderer)
    if icversion is None:
        return (EEG, "") if return_com else EEG
    output = [await iclabel_async(dataset, algorithm=icversion, engine=engine) for dataset in _datasets_with_ica(EEG)]
    output = output if isinstance(EEG, list) else output[0]
    return (output, _history_command(icversion, asynchronous=True)) if return_com else output


def _requested_icversion(EEG, icversion, gui, renderer):
    """Return the ICLabel version to run, or None when there is no dataset or the dialog was cancelled."""
    if EEG is None:
        return None
    if gui is None:
        gui = icversion is None
    if gui:
        result = _run_gui(renderer=renderer)
        if result is None:
            return None
        icversion = result["icversion"]
    icversion = "default" if icversion is None else str(icversion).lower()
    if icversion not in _VERSIONS:
        raise ValueError("icversion must be one of 'default', 'lite', or 'beta'")
    return icversion


def _datasets_with_ica(EEG):
    """Return ``EEG`` as a list of datasets after checking that each one has ICA weights."""
    datasets = EEG if isinstance(EEG, list) else [EEG]
    for dataset in datasets:
        _require_ica(dataset)
    return datasets


def pop_iclabel_dialog_spec() -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_iclabel``."""
    return DialogSpec(
        title="ICLabel",
        function_name="pop_iclabel",
        eeglab_source="plugins/ICLabel/pop_iclabel.m",
        geometry=((1,), (1,)),
        size=(356, 199),
        show_help_button=False,
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
    if weights is None or np.asarray(weights).size == 0:
        raise ValueError("ICLabel requires an ICA decomposition. Run pop_runica first.")


def _history_command(icversion, *, asynchronous=False):
    if asynchronous:
        return f"EEG = await pop_iclabel_async(EEG, '{icversion}');"
    return f"EEG = pop_iclabel(EEG, '{icversion}');"

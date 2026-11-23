"""Import a Neuroscan epoch ``.eeg`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_importepoch import pop_importepoch
from eegprep.functions.sigprocfunc.loadeeg import loadeeg


def pop_loadeeg(
    filename: str | Path,
    *,
    gui: bool = False,
    renderer: Any = None,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Load a Neuroscan ``.eeg`` epoch file the way ``pop_loadeeg.m`` does.

    The dialog asks for precision and trial, type, electrode, and response
    subsets. Cancel leaves the caller unchanged. Events are built with
    ``pop_importepoch`` so each sweep has a time-locking event and, when
    reaction times are present, an RT event.
    """
    path = Path(filename)
    if not path.is_file():
        raise ValueError(
            f"pop_loadeeg: {path} is not a Neuroscan .eeg file. Choose a .eeg file with File > Import data."
        )
    ranges: dict[str, Any] = {}
    if gui or renderer is not None:
        result = inputgui(pop_loadeeg_dialog_spec(), renderer=renderer)
        if not result:
            return ({}, "") if return_com else {}
        ranges = _ranges(result)
    signal, accept, types, rts, responses, names, points, sweeps, srate, xmin, xmax = loadeeg(path, **ranges)
    if sweeps < 1 or points < 1:
        raise ValueError(
            f"pop_loadeeg: {path} has no sweeps to import. Check the trial and type ranges, then import again."
        )
    n_channels = int(signal.shape[0])
    data = np.transpose(signal.reshape(n_channels, sweeps, points), (0, 2, 1))
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": "Neuroscan EEG data",
            "comments": f"Original file: {path}",
            "filename": path.name,
            "filepath": str(path.parent),
            "nbchan": n_channels,
            "pnts": int(points),
            "trials": int(sweeps),
            "srate": float(srate),
            "xmin": float(xmin),
            "xmax": float(xmax),
            "data": data,
            "chanlocs": [{"labels": name} for name in names],
        }
    )
    fields = ["type", "accept", "response"]
    columns = [types, accept, responses]
    latency_fields: list[str] = []
    if np.any(np.asarray(rts)):
        fields = ["RT", *fields]
        columns = [np.asarray(rts, dtype=float) * 1000.0, *columns]
        latency_fields = ["RT"]
    table = np.column_stack(columns)
    eeg = pop_importepoch(
        eeg,
        table,
        fields,
        latencyfields=latency_fields,
        timeunit=1e-3,
        headerlines=0,
        clearevents=1,
    )
    command = f"EEG = pop_loadeeg({format_history_value(path)});"
    history = str(eeg.get("history") or "").strip()
    eeg["history"] = f"{history}\n{command}".lstrip()
    return (eeg, command) if return_com else eeg


def pop_loadeeg_dialog_spec() -> DialogSpec:
    """Return the precision and subset dialog from ``pop_loadeeg.m``."""
    return DialogSpec(
        title="Import Neuroscan EEG -- pop_loadeeg()",
        function_name="pop_loadeeg",
        help_text="pop_loadeeg",
        eeglab_source="plugins/neuroscanio/pop_loadeeg.m",
        geometry=((1, 2), (1, 2), (1, 2), (1, 2), (1, 2)),
        controls=(
            ControlSpec("text", "Data precision"),
            ControlSpec("popupmenu", "Auto|16-bit|32-bit", tag="precision", value=1),
            ControlSpec("text", "Trial subset (default all)"),
            ControlSpec("edit", tag="trials"),
            ControlSpec("text", "Type subset (default all)"),
            ControlSpec("edit", tag="types"),
            ControlSpec("text", "Electrode subset (default all)"),
            ControlSpec("edit", tag="channels"),
            ControlSpec("text", "Response subset (default all)"),
            ControlSpec("edit", tag="responses"),
        ),
    )


def _ranges(result: dict[str, Any]) -> dict[str, Any]:
    ranges: dict[str, Any] = {}
    precision = int(result.get("precision") or 1)
    if precision == 2:
        ranges["format"] = "short"
    elif precision == 3:
        ranges["format"] = "int32"
    for tag, key in (
        ("trials", "triallist"),
        ("types", "typerange"),
        ("channels", "chanlist"),
        ("responses", "responsetype"),
    ):
        text = str(result.get(tag) or "").strip()
        if text:
            ranges[key] = text
    return ranges

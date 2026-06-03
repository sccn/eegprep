"""Template EEGLAB-style pop function."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec

_UNSET = object()


def pop_template_gain(
    EEG: dict[str, Any] | list[dict[str, Any]] | None,
    gain: float | object = _UNSET,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Scale EEG data by ``gain`` and optionally return an EEGLAB-style command."""
    if EEG is None:
        return (None, "") if return_com else None

    if gui is None:
        gui = gain is _UNSET
    if gui:
        result = _run_gui(renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        gain = result["gain"]
    elif gain is _UNSET:
        gain = 1.0

    gain_value = _validated_gain(gain)
    command = _history_command(gain_value)
    if isinstance(EEG, list):
        output = [pop_template_gain(eeg, gain_value, gui=False) for eeg in EEG]
        return (output, command) if return_com else output

    output = deepcopy(EEG)
    data = np.asarray(output.get("data"), dtype=float)
    if data.size == 0:
        raise ValueError("EEG.data must contain samples")
    output["data"] = data * gain_value
    output.setdefault("etc", {})["eegprep_ext_template"] = {"gain": gain_value}
    return (output, command) if return_com else output


def pop_template_gain_dialog_spec() -> DialogSpec:
    """Return the renderer-independent dialog spec for ``pop_template_gain``."""
    return DialogSpec(
        title="Apply template gain -- pop_template_gain()",
        function_name="pop_template_gain",
        eeglab_source="extensions/eegprep_ext_template/pop_template_gain.py",
        geometry=((1,), (1,)),
        size=(330, 180),
        show_help_button=False,
        known_differences=(
            "Extension-packaged help resources are declared in ExtensionSpec; "
            "Dialog Help routing depends on extension runtime integration.",
        ),
        controls=(
            ControlSpec("text", "Multiply EEG.data by this gain"),
            ControlSpec(
                "edit",
                tag="gain",
                value="1",
                callback=CallbackSpec("validate_numeric_range", params={"columns": 1}),
            ),
        ),
    )


def _run_gui(*, renderer: Any | None = None) -> dict[str, float] | None:
    result = inputgui(pop_template_gain_dialog_spec(), renderer=renderer)
    if result is None:
        return None
    return {"gain": _validated_gain(result.get("gain", "1"))}


def _validated_gain(value: Any) -> float:
    gain = float(value)
    if not isfinite(gain):
        raise ValueError("gain must be finite")
    return gain


def _history_command(gain: float) -> str:
    return f"EEG = pop_template_gain(EEG, {_format_number(gain)});"


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"

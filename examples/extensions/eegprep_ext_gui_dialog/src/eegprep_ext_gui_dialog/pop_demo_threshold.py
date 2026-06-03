"""GUI dialog example extension."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec

_UNSET = object()


def pop_demo_threshold(
    EEG: dict[str, Any] | None,
    threshold: float | object = _UNSET,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Record how many samples exceed a threshold, with an optional GUI."""
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = threshold is _UNSET
    if gui:
        result = inputgui(pop_demo_threshold_dialog_spec(), renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        threshold = result.get("threshold", "0")
    elif threshold is _UNSET:
        threshold = 0.0

    threshold_value = float(threshold)
    output = deepcopy(EEG)
    count = int(np.count_nonzero(np.asarray(output["data"], dtype=float) > threshold_value))
    output.setdefault("etc", {})["eegprep_ext_gui_dialog"] = {
        "threshold": threshold_value,
        "samples_over_threshold": count,
    }
    command = f"EEG = pop_demo_threshold(EEG, {threshold_value:g});"
    return (output, command) if return_com else output


def pop_demo_threshold_dialog_spec() -> DialogSpec:
    """Return the renderer-independent dialog spec for the GUI example."""
    return DialogSpec(
        title="Example threshold -- pop_demo_threshold()",
        function_name="pop_demo_threshold",
        eeglab_source="extensions/eegprep_ext_gui_dialog/pop_demo_threshold.py",
        geometry=((1,), (1,)),
        size=(330, 180),
        show_help_button=False,
        known_differences=(
            "Extension dialog Help routing should be enabled only after the runtime "
            "can resolve extension-packaged help resources.",
        ),
        controls=(
            ControlSpec("text", "Threshold"),
            ControlSpec(
                "edit",
                tag="threshold",
                value="0",
                callback=CallbackSpec("validate_numeric_range", params={"columns": 1}),
            ),
        ),
    )

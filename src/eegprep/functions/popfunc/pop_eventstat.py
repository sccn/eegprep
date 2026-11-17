"""EEGLAB-style wrapper for event-field statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import history_command, numeric_vector, show_figures
from eegprep.functions.popfunc._pop_utils import is_empty_value as _is_empty
from eegprep.functions.popfunc.eeg_getepochevent import eeg_getepochevent
from eegprep.functions.sigprocfunc.signalstat import signalstat


def pop_eventstat(
    EEG: dict[str, Any] | None = None,
    eventfield: str | None = None,
    type: Any = None,
    latrange: Any = None,
    percent: float = 5,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    return_com: bool = False,
):
    """Compute and plot statistics for numeric EEG event fields.

    Pass ``plot='off'`` to build and return the figure without opening a window.
    """
    if EEG is None:
        return (None, "") if return_com else None
    if gui is None:
        gui = eventfield is None
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (None, "") if return_com else None
        eventfield = result["eventfield"]
        type = result["type"]
        latrange = result["latrange"]
        percent = result["percent"]
    eventfield = str(eventfield or "latency")
    values = event_values(EEG, eventfield, type=type, latrange=latrange)
    if values.size == 0:
        raise ValueError("No such events found. See Edit > Event values to confirm event type.")
    label = "Event values"
    title = (
        f"All event statistics for '{eventfield}' info"
        if _is_empty(type)
        else f"Event {type!r} statistics for '{eventfield}' info"
    )
    result = signalstat(values, 1, label, float(percent), title)
    command = history_command("pop_eventstat", eventfield, type, latrange, percent)
    show_figures(result.figure, plot=plot)
    return (result, command) if return_com else result


def pop_eventstat_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_eventstat``."""
    _ = EEG
    controls = (
        ControlSpec("text", "Event field to process:"),
        ControlSpec("edit", tag="eventfield", value="latency"),
        ControlSpec("text", 'Event type(s) ([]=all):\nSelect "Edit > Event values" to see type values'),
        ControlSpec("edit", tag="type", value=""),
        ControlSpec("text", "Event latency range (ms)\nDefault is whole epoch or data"),
        ControlSpec("edit", tag="latrange", value=""),
        ControlSpec("text", "Percent for trimmed statistics:"),
        ControlSpec("edit", tag="percent", value="5"),
    )
    return DialogSpec(
        title="Plot event statistics -- pop_eventstat()",
        controls=controls,
        geometry=((1, 1), (1, 1), (1, 1), (1, 1)),
        function_name="pop_eventstat",
        eeglab_source="functions/popfunc/pop_eventstat.m",
        help_text="pophelp('pop_eventstat')",
        size=(560, 230),
    )


def event_values(EEG: dict[str, Any], eventfield: str, *, type: Any = None, latrange: Any = None) -> np.ndarray:
    """Extract numeric event-field values using EEGLAB-style filters.

    Latency and duration come back in milliseconds, the same conversion
    ``eeg_getepochevent`` applies.
    """
    requested_types = [] if _is_empty(type) else list(_type_set(type))
    bounds = numeric_vector(latrange)
    timewin = bounds.tolist() if bounds.size == 2 else [-np.inf, np.inf]
    _epoch_values, grouped = eeg_getepochevent(EEG, requested_types, timewin, eventfield)
    values = []
    for group in grouped:
        for item in group:
            try:
                number = float(np.asarray(item).squeeze())
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append(number)
    return np.asarray(values, dtype=float)


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_eventstat_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    return {
        "eventfield": str(result.get("eventfield", "latency") or "latency").strip(),
        "type": _parse_type_text(result.get("type", "")),
        "latrange": numeric_vector(result.get("latrange", [])).tolist(),
        "percent": float(numeric_vector(result.get("percent", 5))[0]),
    }


def _type_set(value: Any) -> set[str]:
    if _is_empty(value):
        return set()
    if isinstance(value, str):
        return {part.strip().strip("'\"") for part in value.replace(",", " ").split() if part.strip()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {str(item).strip().strip("'\"") for item in np.asarray(list(value), dtype=object).ravel().tolist()}
    return {str(value)}


def _parse_type_text(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return []
    return sorted(_type_set(text))


__all__ = ["event_values", "pop_eventstat", "pop_eventstat_dialog_spec"]

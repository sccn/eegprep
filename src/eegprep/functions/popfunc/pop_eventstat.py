"""EEGLAB-style wrapper for event-field statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc.plot_utils import history_command, numeric_vector, show_figures
from eegprep.functions.popfunc._pop_utils import is_empty_value as _is_empty
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
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
    return_com: bool = False,
):
    """Compute and plot statistics for numeric EEG event fields."""
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
    show_figures(result.figure)
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
    """Extract numeric event-field values using EEGLAB-style filters."""
    requested_types = _type_set(type)
    lat_bounds = numeric_vector(latrange)
    values = []
    for event in _event_list(EEG.get("event", [])):
        if requested_types and str(event.get("type", "")) not in requested_types:
            continue
        if lat_bounds.size == 2 and not _event_in_latency_range(EEG, event, lat_bounds):
            continue
        if eventfield not in event:
            continue
        try:
            value = float(np.asarray(event[eventfield]).squeeze())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
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


def _event_list(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.ravel().tolist()
    if isinstance(events, dict):
        return [events]
    return [event for event in events if isinstance(event, dict)]


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


def _event_in_latency_range(EEG: dict[str, Any], event: dict[str, Any], bounds: np.ndarray) -> bool:
    try:
        latency = float(np.asarray(event.get("latency")).squeeze())
    except (TypeError, ValueError):
        return False
    try:
        epoch = float(np.asarray(event.get("epoch", 1)).squeeze())
    except (TypeError, ValueError):
        epoch = 1.0
    latency_ms = float(
        eeg_point2lat(
            [latency],
            [epoch],
            float(EEG.get("srate", 1) or 1),
            [float(EEG.get("xmin", 0) or 0) * 1000.0, float(EEG.get("xmax", 0) or 0) * 1000.0],
            1e-3,
        )[0]
    )
    return bool(bounds[0] <= latency_ms <= bounds[1])


__all__ = ["event_values", "pop_eventstat", "pop_eventstat_dialog_spec"]

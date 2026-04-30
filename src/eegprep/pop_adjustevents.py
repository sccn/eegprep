"""Adjust EEGLAB event latencies."""

from __future__ import annotations

from copy import deepcopy
import re
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np

from eegprep.eeg_checkset import eeg_checkset
from eegprep.eeg_options import EEG_OPTIONS


def pop_adjustevents(
    EEG: dict[str, Any],
    *,
    addms: float | None = None,
    addsamples: float | None = None,
    eventtypes: str | Iterable[object] | None = None,
    eventtype: str | Iterable[object] | None = None,
    force: str | bool = "auto",
    gui: bool = False,
    renderer: Any | None = None,
) -> tuple[dict[str, Any], str]:
    """Adjust event latencies by milliseconds or samples.

    This ports EEGLAB ``pop_adjustevents``. Event latencies remain EEGLAB-style
    1-based floating sample positions.

    Parameters
    ----------
    EEG
        EEGLAB-style EEG dictionary.
    addms
        Milliseconds to add to matching event latencies.
    addsamples
        Samples to add to matching event latencies.
    eventtypes, eventtype
        Event type or event types to shift. Empty/``None`` means all events.
    force
        ``"on"``, ``"off"``, or ``"auto"``. ``"auto"`` matches EEGLAB:
        GUI calls default to ``"off"`` and command-line calls default to ``"on"``.
    gui
        If true, collect options using the optional GUI renderer.
    renderer
        Optional test/custom renderer for GUI specs.

    Returns
    -------
    EEG, com
        Adjusted EEG dictionary and EEGLAB-style history command.
    """

    force_was_explicit = not (isinstance(force, str) and force.lower() == "auto")
    if eventtypes is None and eventtype is not None:
        eventtypes = eventtype

    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return EEG, ""
        addms, addsamples, eventtypes, force = result
        force_was_explicit = force == "on"

    options_for_com: list[tuple[str, Any]] = []
    if addms is not None:
        options_for_com.append(("addms", addms))
    elif addsamples is not None:
        options_for_com.append(("addsamples", addsamples))
    else:
        raise ValueError("To adjust event latencies, specify addms or addsamples")

    force_mode = _normalize_force(force, gui=gui)
    if force_was_explicit:
        options_for_com.append(("force", force_mode))

    parsed_eventtypes = _parse_eventtypes(eventtypes)
    if parsed_eventtypes:
        options_for_com.append(("eventtypes", parsed_eventtypes))

    EEG_out = deepcopy(EEG)
    events = _event_list(EEG_out)
    if not events:
        raise ValueError("Unable to proceed. No events found")

    if force_mode == "off":
        _check_force_safe(EEG_out, events)

    indices = _matching_event_indices(events, parsed_eventtypes)
    if not indices:
        raise ValueError("Unable to proceed. Event type(s) requested not found")

    shift = float(addms) / 1000.0 * float(EEG_out["srate"]) if addms is not None else float(addsamples)
    for index in indices:
        if "latency" not in events[index]:
            raise ValueError("Unable to proceed. Event missing latency field")
        events[index]["latency"] = float(events[index]["latency"]) + shift

    EEG_out = eeg_checkset(EEG_out)
    return EEG_out, _format_command(options_for_com)


def _run_gui(
    EEG: dict[str, Any],
    *,
    renderer: Any | None = None,
) -> tuple[float | None, float | None, list[object], str] | None:
    from eegprep.gui import inputgui
    from eegprep.gui.specs import pop_adjustevents_dialog_spec

    spec = pop_adjustevents_dialog_spec(
        float(EEG["srate"]),
        event_types=_unique_event_types(_event_list(EEG)),
    )
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None

    eventtypes = _parse_eventtypes(result.get("events"))
    addms = _optional_float(result.get("edit_time"))
    addsamples = _optional_float(result.get("edit_samples"))
    if addms is None and addsamples is None:
        raise ValueError("Not enough parameters selected")
    force = "on" if bool(result.get("force")) else "off"
    return addms, addsamples, eventtypes, force


def _event_list(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    events = EEG.get("event", [])
    if events is None:
        return []
    if isinstance(events, dict):
        EEG["event"] = [events]
        return EEG["event"]
    if isinstance(events, np.ndarray):
        return list(events)
    return list(events)


def _unique_event_types(events: Iterable[dict[str, Any]]) -> tuple[object, ...]:
    values: list[object] = []
    for event in events:
        value = event.get("type")
        if value not in values:
            values.append(value)
    return tuple(values)


def _normalize_force(force: str | bool, *, gui: bool) -> str:
    if isinstance(force, bool):
        return "on" if force else "off"
    force_mode = str(force).lower()
    if force_mode not in {"on", "off", "auto"}:
        raise ValueError("force must be 'on', 'off', 'auto', True, or False")
    if force_mode == "auto":
        return "off" if gui else "on"
    return force_mode


def _check_force_safe(EEG: dict[str, Any], events: Iterable[dict[str, Any]]) -> None:
    if int(EEG.get("trials", 1)) > 1:
        raise ValueError("Be careful when adjusting latencies for data epochs. Use the 'force' option to do so.")
    if _has_boundary_event(events):
        raise ValueError(
            "Be careful when adjusting latencies when boundary events are present. "
            "Use the 'force' option to do so."
        )


def _has_boundary_event(events: Iterable[dict[str, Any]]) -> bool:
    for event in events:
        event_type = event.get("type")
        if isinstance(event_type, str) and event_type == "boundary":
            return True
        if EEG_OPTIONS.get("option_boundary99") and event_type == 99:
            return True
    return False


def _matching_event_indices(events: list[dict[str, Any]], eventtypes: list[object]) -> list[int]:
    if not eventtypes:
        return list(range(len(events)))

    indices: list[int] = []
    for event_type in eventtypes:
        matches = [
            index
            for index, event in enumerate(events)
            if _event_type_matches(event.get("type"), event_type)
        ]
        if not matches:
            warnings.warn(f"Event type '{event_type}' not found.", RuntimeWarning, stacklevel=2)
        indices.extend(matches)
    return indices


def _event_type_matches(actual: object, requested: object) -> bool:
    return actual == requested or str(actual) == str(requested)


def _parse_eventtypes(value: str | Iterable[object] | None) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text == "{}":
            return []
        quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", text)
        if quoted:
            return [single or double for single, double in quoted]
        return [item for item in re.split(r"[\s,]+", text) if item]
    return [item for item in value if item is not None and item != ""]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _format_command(options: list[tuple[str, Any]]) -> str:
    return f"[EEG,com] = pop_adjustevents(EEG, {_format_options(options)});"


def _format_options(options: list[tuple[str, Any]]) -> str:
    return ", ".join(f"'{key}', {_format_value(value)}" for key, value in options)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_format_value(item) for item in value) + "}"
    return f"{value:g}" if isinstance(value, (int, float)) else repr(value)

"""Add events while preserving event/urevent consistency."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list
from eegprep.functions.popfunc.eeg_urlatency import eeg_urlatency


def eeg_addnewevents(
    EEG: dict[str, Any],
    event_latency_arrays: Any,
    types: Any,
    field_names: Any | None = None,
    field_values: Any | None = None,
) -> dict[str, Any]:
    """Add typed events and rebuild sorted, consistent event structures.

    Each latency array corresponds to one event type. Optional field-value
    arrays span all newly added events in the same concatenated order.
    """
    output = copy.deepcopy(EEG)
    latency_groups = [np.asarray(group, dtype=float).ravel().tolist() for group in event_latency_arrays]
    type_values = [types] if isinstance(types, str) else list(types)
    if len(latency_groups) != len(type_values):
        raise ValueError("one event type is required for each latency array")
    fields = [] if field_names is None else ([field_names] if isinstance(field_names, str) else list(field_names))
    values = [] if field_values is None else list(field_values)
    total = sum(len(group) for group in latency_groups)
    if len(fields) != len(values):
        raise ValueError("field_names and field_values must have the same length")
    flat_values = [np.asarray(value, dtype=object).ravel().tolist() for value in values]
    if any(len(value) != total for value in flat_values):
        raise ValueError("each field-values array must contain one value per new event")

    original_events = events_as_list(output.get("event", []))
    all_fields = set(fields)
    for event in original_events:
        all_fields.update(event)
    new_events: list[dict[str, Any]] = []
    value_index = 0
    for event_type, latencies in zip(type_values, latency_groups):
        for latency in latencies:
            event = {field: np.nan for field in all_fields}
            event.update({"latency": float(latency), "type": event_type, "duration": 0.0, "urevent": np.nan})
            for field_index, field in enumerate(fields):
                event[field] = flat_values[field_index][value_index]
            new_events.append(event)
            value_index += 1
    for event in original_events:
        event.setdefault("duration", 0.0)
        for field in fields:
            event.setdefault(field, np.nan)
    combined = sorted([*original_events, *new_events], key=lambda event: float(event["latency"]))
    old_urevents = events_as_list(output.get("urevent", []))
    event_urevents: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for event in combined:
        pointer = event.get("urevent")
        if _valid_pointer(pointer, len(old_urevents)):
            original_latency = float(old_urevents[int(pointer)]["latency"])
        else:
            original_latency = float(eeg_urlatency(combined, event["latency"]))
        urevent = {key: copy.deepcopy(value) for key, value in event.items() if key != "urevent"}
        urevent["latency"] = original_latency
        if urevent.get("duration") is None or np.asarray(urevent.get("duration")).size == 0:
            urevent["duration"] = 0.0
        event_urevents.append((original_latency, event, urevent))
    urevents: list[dict[str, Any]] = []
    for pointer, (_latency, event, urevent) in enumerate(sorted(event_urevents, key=lambda item: item[0])):
        event["urevent"] = pointer
        urevents.append(urevent)
    output["event"] = combined
    output["urevent"] = urevents
    return output


def _valid_pointer(value: Any, length: int) -> bool:
    try:
        pointer = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(pointer) and pointer.is_integer() and 0 <= pointer < length


__all__ = ["eeg_addnewevents"]

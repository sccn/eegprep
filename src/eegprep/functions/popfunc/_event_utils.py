"""Shared event helpers for EEGLAB-style pop functions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def events_as_list(events: Any) -> list[dict[str, Any]]:
    """Return EEG events as a mutable list of dictionaries."""
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, dict):
        return [dict(events)]
    return [dict(event) for event in events]


def event_field_names(events: Any, *, include_urevent: bool = False) -> list[str]:
    """Return event field names preserving first-seen order."""
    fields: list[str] = []
    for event in events_as_list(events):
        for field in event:
            if field == "urevent" and not include_urevent:
                continue
            if field not in fields:
                fields.append(field)
    return fields


def normalize_event_indices(indices: Any, length: int, *, allow_empty: bool = False) -> list[int]:
    """Normalize EEGLAB-facing 1-based event indices to Python indices."""
    if indices is None or _is_empty(indices):
        if allow_empty:
            return []
        return list(range(length))
    values = _as_flat_list(indices)
    normalized: list[int] = []
    for value in values:
        index = int(value)
        if index < 1 or index > length:
            raise ValueError("event indices must be 1-based and within EEG.event")
        normalized.append(index - 1)
    return normalized


def normalize_one_based_indices(indices: Any, length: int, *, label: str) -> list[int]:
    """Normalize a required 1-based index vector to Python indices."""
    values = _as_flat_list(indices)
    if not values:
        raise ValueError(f"{label} must contain at least one index")
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        index = int(value)
        if index < 1 or index > length:
            raise ValueError(f"{label} must be 1-based and within range")
        if index not in seen:
            seen.add(index)
            normalized.append(index - 1)
    return normalized


def value_sequence(value: Any, count: int) -> list[Any]:
    """Return ``count`` values, repeating scalars like EEGLAB's setstruct path."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    if len(values) == 1 and count > 1:
        values = values * count
    if len(values) != count:
        raise ValueError("Wrong size for input array")
    return values


def sort_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort events by latency when latency is present."""
    if not events or not any("latency" in event for event in events):
        return events
    return sorted(events, key=lambda event: float(event.get("latency", np.inf)))


def is_boundary_event(event: dict[str, Any]) -> bool:
    """Return true for EEGLAB boundary events."""
    return str(event.get("type", "")).lower() == "boundary"


def event_value_for_history(value: Any) -> Any:
    """Normalize event values before history formatting."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _as_flat_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (str, bytes)):
        return [int(token) for token in str(value).strip().strip("[]").replace(",", " ").split() if token]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) == 0
    return False

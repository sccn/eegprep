"""Boundary event type selected from the first event."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS


def eeg_boundarytype(first: Any, second: Any | None = None) -> str | int:
    """Return the boundary event type for one or two event lists.

    ``eeg_boundarytype.m`` returns ``-99`` when ``option_boundary99`` is on
    and the first event type of each input is numeric. An empty list counts
    as numeric. Two empty inputs stay ``'boundary'``. An EEG dict is read
    through its ``event`` field when it also has ``setname``.
    """
    if second is None:
        second = first
    events1 = _boundary_events(first)
    events2 = _boundary_events(second)
    if not EEG_OPTIONS["option_boundary99"] or (not events1 and not events2):
        return "boundary"
    if _first_type_is_numeric(events1) and _first_type_is_numeric(events2):
        return -99
    return "boundary"


def _boundary_events(value: Any) -> list[Any]:
    if isinstance(value, dict) and "event" in value and "setname" in value:
        value = value.get("event")
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    return list(value)


def _first_type_is_numeric(events: list[Any]) -> bool:
    if not events:
        return True
    event = events[0]
    if isinstance(event, np.ndarray) and event.shape == ():
        event = event.item()
    if not isinstance(event, dict):
        return False
    event_type = event.get("type")
    return isinstance(event_type, (int, float, np.integer, np.floating))

"""Shared EEGLAB-style event helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS


def boundary_event_indices(EEG: Any) -> list[int]:
    """Return 0-based indices of EEGLAB boundary events."""
    events = _event_records(EEG)
    if not events or not isinstance(events[0], dict) or "type" not in events[0]:
        return []
    first_type = events[0].get("type")
    if isinstance(first_type, bytes):
        first_type = first_type.decode("utf-8")
    if isinstance(first_type, str):
        return [
            index for index, event in enumerate(events) if _string_event_type(event.get("type")).startswith("boundary")
        ]
    if EEG_OPTIONS["option_boundary99"]:
        return [index for index, event in enumerate(events) if _numeric_event_type(event.get("type")) == -99]
    return []


def is_boundary_event(event: dict[str, Any]) -> bool:
    """Return whether one event matches EEGLAB boundary semantics."""
    return bool(boundary_event_indices([event]))


def _event_records(EEG: Any) -> list[Any]:
    if EEG is None:
        return []
    if isinstance(EEG, dict) and "event" in EEG and "setname" in EEG:
        EEG = EEG["event"]
    if isinstance(EEG, np.ndarray):
        EEG = EEG.tolist()
    if isinstance(EEG, dict):
        return [EEG]
    if isinstance(EEG, (list, tuple)):
        return list(EEG)
    return []


def _string_event_type(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return ""


def _numeric_event_type(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None

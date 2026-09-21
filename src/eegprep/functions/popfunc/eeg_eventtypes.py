"""Summarize event types in an EEG dataset."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list


def eeg_eventtypes(EEG: dict[str, Any]) -> tuple[list[str], list[int]]:
    """Return event type names and counts, sorted by decreasing count.

    Ties follow EEGLAB's reverse-alphabetical ordering.
    """
    if not isinstance(EEG, dict):
        raise TypeError("EEG must be a dataset dictionary")
    if "event" not in EEG:
        raise ValueError("EEG.event field not found")
    counts = Counter(_type_text(event.get("type", "")) for event in events_as_list(EEG["event"]))
    ordered = sorted(counts, key=lambda value: (counts[value], value), reverse=True)
    return ordered, [counts[value] for value in ordered]


def _type_text(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


__all__ = ["eeg_eventtypes"]

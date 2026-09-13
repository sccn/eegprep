"""Event-field value and histogram summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list


def eeg_eventhist(events: Any, field: str, bins: int | Any = 10) -> tuple[list[Any] | np.ndarray, np.ndarray, Any]:
    """Return event-field values, histogram counts, and bin labels or edges."""
    event_list = events_as_list(events)
    if not event_list:
        raise ValueError("event structure is empty")
    if not any(field in event for event in event_list):
        raise ValueError(f"named field {field!r} is not an event field")
    first = next((event.get(field) for event in event_list if _has_value(event.get(field))), None)
    if first is None:
        raise ValueError(f"all event fields named {field!r} are empty")

    if isinstance(first, str):
        values = [str(event.get(field) or " ") for event in event_list]
        labels = sorted(set(values))
        counts = np.asarray([values.count(label) for label in labels], dtype=int)
        return values, counts, labels
    if isinstance(first, dict):
        return [event.get(field) for event in event_list], np.array([], dtype=int), []

    values = np.asarray(
        [float(event[field]) if _has_value(event.get(field)) else np.nan for event in event_list],
        dtype=float,
    )
    if np.isscalar(bins):
        number = int(bins)
        if number < 3:
            raise ValueError("number of bins must be greater than 2")
        mean = float(np.nanmean(values))
        standard_deviation = float(np.nanstd(values, ddof=1)) if np.count_nonzero(np.isfinite(values)) > 1 else 0.0
        offsets = np.arange(-(number // 2), int(np.ceil(number / 2)) + 1)
        edges = mean + offsets * standard_deviation
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        edges = np.asarray(bins, dtype=float)
    finite_values = values[np.isfinite(values)]
    counts = np.zeros(len(edges) - 1, dtype=int)
    for index in range(len(counts)):
        mask = (finite_values >= edges[index]) & (finite_values < edges[index + 1])
        counts[index] = int(np.count_nonzero(mask))
    return values, counts, edges


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple)):
        return bool(value)
    return True


__all__ = ["eeg_eventhist"]

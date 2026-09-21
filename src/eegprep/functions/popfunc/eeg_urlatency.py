"""Recover original sample latencies across boundary events."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list, is_boundary_event


def eeg_urlatency(events: Any, latencies: Any) -> float | np.ndarray:
    """Map current-data latencies to their original continuous positions.

    Boundary durations are added when the boundary precedes an input latency,
    matching EEGLAB's 1-based, fractional-sample latency convention.
    """
    input_array = np.asarray(latencies, dtype=float)
    output = input_array.copy()
    boundary_events = [event for event in events_as_list(events) if is_boundary_event(event)]
    if boundary_events and any("duration" not in event for event in boundary_events):
        output[...] = np.nan
    else:
        for event in boundary_events:
            duration = event.get("duration")
            if duration is None or np.asarray(duration).size == 0:
                output[...] = np.nan
                break
            output = np.where(float(event["latency"]) < input_array, output + float(duration), output)
    return float(output) if output.ndim == 0 else output


__all__ = ["eeg_urlatency"]

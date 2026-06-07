"""FIRFilt boundary-latency helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._file_io import events_to_records


def findboundaries(event: Any) -> np.ndarray:
    """Return EEGLAB FIRFilt boundary latencies as 1-based sample onsets."""
    events = events_to_records(event)
    boundaries: list[int] = []
    if events and all("type" in record and "latency" in record for record in events):
        first_type = events[0].get("type")
        for record in events:
            event_type = record.get("type")
            is_boundary = (
                isinstance(first_type, str) and isinstance(event_type, str) and event_type.startswith("boundary")
            ) or (not isinstance(first_type, str) and event_type == -99)
            if not is_boundary:
                continue
            try:
                boundaries.append(int(np.fix(float(record.get("latency")) + 0.5)))
            except (TypeError, ValueError):
                continue
    unique = sorted(set(boundaries))
    if not unique or unique[0] != 1:
        unique.insert(0, 1)
    return np.asarray(unique, dtype=int)

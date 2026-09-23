"""Insert boundary events after removing continuous sample regions."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.miscfunc.misc import round_mat
from eegprep.functions.popfunc._event_utils import events_as_list, is_boundary_event


def eeg_insertbound(
    events: Any,
    pnts: int,
    regions: Any,
    lengths: Any | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Insert boundaries and adjust event latencies for removed regions.

    Regions use EEGLAB's 1-based inclusive ``[begin, end]`` convention. The
    optional ``lengths`` argument is accepted for API compatibility; current
    EEGLAB derives lengths from the regions themselves. Returned event indices
    are 0-based for direct use with the Python event list.
    """
    del lengths
    output = [copy.deepcopy(event) for event in events_as_list(events)]
    region_array = np.asarray(regions, dtype=float)
    if region_array.size == 0:
        return output, []
    region_array = np.atleast_2d(round_mat(region_array).astype(int))
    if region_array.shape[1] != 2:
        raise ValueError("regions must have shape (n, 2)")
    region_array = np.clip(region_array, 1, int(pnts))
    region_array = region_array[np.argsort(region_array[:, 0])]
    if np.any(region_array[:, 0] > region_array[:, 1]):
        raise ValueError("each region must satisfy begin <= end")
    for index in range(1, len(region_array)):
        if region_array[index - 1, 1] >= region_array[index, 0]:
            region_array[index, 0] = region_array[index - 1, 1] + 1
    region_array = region_array[region_array[:, 0] <= region_array[:, 1]]
    durations = region_array[:, 1] - region_array[:, 0] + 1
    original_count = len(output)
    extra_fields = (
        set().union(*(event.keys() for event in output)) - {"type", "latency", "duration"} if output else set()
    )
    original_latencies = np.asarray([float(event["latency"]) for event in output], dtype=float)
    adjusted_latencies = original_latencies.copy()
    remove: set[int] = set()

    for region_index, (begin, end) in enumerate(region_array):
        adjusted_latencies[original_latencies > begin] -= durations[region_index]
        interior = np.flatnonzero((original_latencies > begin) & (original_latencies < end))
        extra_duration = sum(
            float(output[index].get("duration", 0) or 0) for index in interior if is_boundary_event(output[index])
        )
        remove.update(interior.tolist())
        boundary = {field: np.array([]) for field in extra_fields}
        boundary.update(
            {
                "type": _boundary_type(output[:original_count]),
                "latency": float(begin - np.sum(durations[:region_index]) - 0.5),
                "duration": float(durations[region_index] + extra_duration),
                "_eegprep_new_boundary": True,
            }
        )
        output.append(boundary)

    retained = []
    for index, event in enumerate(output[:original_count]):
        if index not in remove:
            event["latency"] = float(adjusted_latencies[index])
            if event["latency"] >= 0:
                retained.append(event)
    retained.extend(output[original_count:])
    retained.sort(key=lambda event: float(event["latency"]))
    new_indices: list[int] = []
    for index, event in enumerate(retained):
        if event.pop("_eegprep_new_boundary", False):
            new_indices.append(index)
    return retained, new_indices


def _boundary_type(events: list[dict[str, Any]]) -> str | int:
    numeric = not events or isinstance(events[0].get("type"), (int, float, np.integer, np.floating))
    if numeric and EEG_OPTIONS["option_boundary99"] and events:
        return -99
    return "boundary"


__all__ = ["eeg_insertbound"]

"""Merge any number of channel-location montages."""

from __future__ import annotations

import warnings
from typing import Any

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.eeg_mergechan import eeg_mergechan


def eeg_mergelocs(*location_sets: Any) -> tuple[list[dict[str, Any]], bool]:
    """Merge ordered montages and report incompatible electrode ordering."""
    if not location_sets:
        return [], False
    ordered = sorted((chanlocs_as_list(locations) for locations in location_sets), key=len, reverse=True)
    merged = ordered[0]
    incompatible = False
    for locations in ordered[1:]:
        candidate = eeg_mergechan(merged, locations)
        unique_labels = {str(location.get("labels", "")).lower() for location in (*merged, *locations)}
        if len(candidate) > len(unique_labels):
            incompatible = True
            present = {str(location.get("labels", "")).lower() for location in merged}
            candidate = [
                *merged,
                *(location for location in locations if str(location.get("labels", "")).lower() not in present),
            ]
        merged = candidate
    if incompatible:
        warnings.warn(
            "different channel montage or electrode order for the datasets",
            RuntimeWarning,
            stacklevel=2,
        )
    return merged, incompatible


__all__ = ["eeg_mergelocs"]

"""Merge two ordered channel-location sequences."""

from __future__ import annotations

import copy
from typing import Any

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def eeg_mergechan(locations1: Any, locations2: Any) -> list[dict[str, Any]]:
    """Merge channel locations by label while preserving montage order."""
    first = [copy.deepcopy(location) for location in chanlocs_as_list(locations1)]
    second = [copy.deepcopy(location) for location in chanlocs_as_list(locations2)]
    labels1 = [str(location.get("labels", "")).lower() for location in first]
    labels2 = [str(location.get("labels", "")).lower() for location in second]
    merged: list[dict[str, Any]] = []
    index1 = index2 = 0
    while index1 < len(first) or index2 < len(second):
        if index1 >= len(first):
            merged.append(second[index2])
            index2 += 1
        elif index2 >= len(second):
            merged.append(first[index1])
            index1 += 1
        elif labels1[index1] == labels2[index2]:
            merged.append(first[index1])
            index1 += 1
            index2 += 1
        elif labels1[index1] not in labels2:
            merged.append(first[index1])
            index1 += 1
        else:
            merged.append(second[index2])
            index2 += 1
    return merged


__all__ = ["eeg_mergechan"]

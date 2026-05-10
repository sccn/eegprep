"""Store EEG datasets in an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset


def _normalize_index(index: int | None, alleeg: list[dict[str, Any]]) -> int:
    if index is None or index == 0:
        return len(alleeg) + 1
    if index < 1:
        raise ValueError("EEGLAB dataset indices are 1-based")
    return int(index)


def eeg_store(
    ALLEEG: list[dict[str, Any]] | None,
    EEG: dict[str, Any] | list[dict[str, Any]],
    storeSetIndex: int | list[int] | tuple[int, ...] | None = None,
    *_args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]], int | list[int]]:
    """Store EEG in ``ALLEEG`` using EEGLAB-facing 1-based indices."""
    alleeg = [] if ALLEEG is None else list(ALLEEG)

    if isinstance(EEG, list):
        indices = list(storeSetIndex) if isinstance(storeSetIndex, (list, tuple)) else [None] * len(EEG)
        if len(indices) != len(EEG):
            raise ValueError("Length of EEG list must equal length of storeSetIndex")
        stored_indices: list[int] = []
        stored_eeg: list[dict[str, Any]] = []
        for dataset, index in zip(EEG, indices):
            alleeg, checked, stored_index = eeg_store(alleeg, dataset, index)
            stored_eeg.append(checked)
            stored_indices.append(int(stored_index))
        return alleeg, stored_eeg, stored_indices

    checked = eeg_checkset(deepcopy(EEG))
    index = _normalize_index(storeSetIndex if isinstance(storeSetIndex, int) else None, alleeg)
    while len(alleeg) < index:
        alleeg.append({})
    saved_state = str(checked.get("saved") or "").lower()
    is_loaded = saved_state == "justloaded"
    checked["saved"] = "yes" if is_loaded else "no"
    alleeg[index - 1] = checked
    return alleeg, checked, index

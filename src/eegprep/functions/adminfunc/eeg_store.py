"""Store EEG datasets in an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.storage import offload_storedisk_datasets
from eegprep.functions.miscfunc.matlab_inputs import as_dataset_list, matlab_ints, struct_records


def _datasets_or_one(eeg: Any) -> Any:
    records = _dataset_records(eeg)
    if records is None:
        return eeg
    if len(records) == 1:
        return records[0]
    return records


def _dataset_records(eeg: Any) -> list[dict[str, Any]] | None:
    if isinstance(eeg, np.ndarray) and eeg.dtype.names:
        return struct_records(eeg) or []
    if isinstance(eeg, np.ndarray) and eeg.dtype == object and eeg.size:
        items = list(eeg.ravel(order="F"))
        if all(isinstance(item, dict) for item in items):
            return items
    return None


def _normalize_store_index(index: Any) -> int | list[int] | None:
    if index is None or isinstance(index, int):
        return index
    ints = matlab_ints(index)
    if ints is None:
        raise TypeError("eeg_store index must be an integer or a vector of integers")
    if len(ints) == 1:
        return ints[0]
    return ints


def _normalize_index(index: int | None, alleeg: list[dict[str, Any]]) -> int:
    if index is None or index == 0:
        # EEGLAB stores a new dataset in the lowest empty slot (here ``{}``) before appending.
        for position, dataset in enumerate(alleeg, start=1):
            if not dataset:
                return position
        return len(alleeg) + 1
    if index < 1:
        raise ValueError("EEGLAB dataset indices are 1-based")
    return int(index)


def eeg_store(
    ALLEEG: list[dict[str, Any]] | None,
    EEG: dict[str, Any] | list[dict[str, Any]],
    storeSetIndex: Any = None,
    *_args: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | list[dict[str, Any]], int | list[int]]:
    """Store EEG in ``ALLEEG`` using EEGLAB-facing 1-based indices.

    ``ALLEEG`` and ``EEG`` may be MATLAB struct arrays of datasets. A
    whole-number float or a 1-by-1 numeric array is one index, and a numeric
    vector is one index per dataset. One dataset stored from a 1-by-1 struct
    uses that index. The stored collection is still a Python list of dataset
    dicts.
    """
    alleeg = as_dataset_list(ALLEEG)
    EEG = _datasets_or_one(EEG)
    index = _normalize_store_index(storeSetIndex)
    if isinstance(EEG, list) and len(EEG) == 1 and isinstance(index, int):
        EEG = EEG[0]

    if isinstance(EEG, list):
        indices = index if isinstance(index, list) else [None] * len(EEG)
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
    index = _normalize_index(index if isinstance(index, int) else None, alleeg)
    while len(alleeg) < index:
        alleeg.append({})
    saved_state = str(checked.get("saved") or "").lower()
    is_loaded = saved_state == "justloaded"
    checked["saved"] = "yes" if is_loaded else "no"
    alleeg[index - 1] = checked
    offload_storedisk_datasets(alleeg, {index})
    return alleeg, checked, index

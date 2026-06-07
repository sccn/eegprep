"""Retrieve datasets from an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.adminfunc.storage import dataset_with_loaded_data, offload_storedisk_datasets, storedisk_enabled
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def eeg_retrieve(
    ALLEEG: list[dict[str, Any]] | None,
    index: int | list[int] | tuple[int, ...],
) -> tuple[dict[str, Any] | list[dict[str, Any]], list[dict[str, Any]], int | list[int]]:
    """Return dataset(s) from ``ALLEEG`` using EEGLAB-facing 1-based indices."""
    alleeg = [] if ALLEEG is None else list(ALLEEG)
    if isinstance(index, (list, tuple)):
        indices = [int(item) for item in index]
        if storedisk_enabled():
            offload_storedisk_datasets(alleeg, set())
        datasets = [deepcopy(_dataset_at(alleeg, item)) for item in indices]
        return datasets, alleeg, indices
    current = int(index)
    dataset = dataset_with_loaded_data(_dataset_at(alleeg, current))
    if 1 <= current <= len(alleeg):
        alleeg[current - 1] = deepcopy(dataset)
    offload_storedisk_datasets(alleeg, {current})
    return dataset, alleeg, current


def _dataset_at(alleeg: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index < 1:
        raise ValueError("EEGLAB dataset indices are 1-based")
    try:
        dataset = alleeg[index - 1]
    except IndexError as exc:
        raise IndexError(f"No dataset at EEGLAB index {index}") from exc
    return dataset if dataset else eeg_emptyset()

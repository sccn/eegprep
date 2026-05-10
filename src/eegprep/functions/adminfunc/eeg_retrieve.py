"""Retrieve datasets from an EEGLAB-like ALLEEG list."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def eeg_retrieve(
    ALLEEG: list[dict[str, Any]] | None,
    index: int | list[int] | tuple[int, ...],
) -> tuple[dict[str, Any] | list[dict[str, Any]], list[dict[str, Any]], int | list[int]]:
    """Return dataset(s) from ``ALLEEG`` using EEGLAB-facing 1-based indices."""
    alleeg = [] if ALLEEG is None else list(ALLEEG)
    if isinstance(index, (list, tuple)):
        datasets = [deepcopy(_dataset_at(alleeg, int(item))) for item in index]
        return datasets, alleeg, [int(item) for item in index]
    dataset = deepcopy(_dataset_at(alleeg, int(index)))
    return dataset, alleeg, int(index)


def _dataset_at(alleeg: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if index < 1:
        raise ValueError("EEGLAB dataset indices are 1-based")
    try:
        dataset = alleeg[index - 1]
    except IndexError as exc:
        raise IndexError(f"No dataset at EEGLAB index {index}") from exc
    return dataset if dataset else eeg_emptyset()

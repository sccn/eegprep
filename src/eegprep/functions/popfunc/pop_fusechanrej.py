"""Fuse channel rejection across matching subject/session datasets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eegprep.functions.popfunc.pop_select import pop_select


def pop_fusechanrej(
    EEG: list[dict[str, Any]] | dict[str, Any] | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
):
    """Keep only common channels across datasets with the same subject/session."""
    del args, kwargs
    if EEG is None:
        return (None, "") if return_com else None
    if isinstance(EEG, dict):
        return (EEG, "") if return_com else EEG
    if len(EEG) <= 1:
        return (EEG, "") if return_com else EEG
    if any(not dataset.get("subject") for dataset in EEG):
        command = "ALLEEG = pop_fusechanrej(ALLEEG);"
        return (EEG, command) if return_com else EEG
    groups: dict[tuple[str, Any], list[int]] = defaultdict(list)
    for index, dataset in enumerate(EEG):
        groups[(str(dataset.get("subject")), dataset.get("session"))].append(index)
    output = list(EEG)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        common = _common_channel_labels([output[index] for index in indices])
        for index in indices:
            labels = _channel_labels(output[index])
            if labels != common:
                output[index] = pop_select(output[index], "channel", common, gui=False)
    command = "ALLEEG = pop_fusechanrej(ALLEEG);"
    return (output, command) if return_com else output


def _channel_labels(EEG: dict[str, Any]) -> list[str]:
    return [str(chan.get("labels", "")) for chan in EEG.get("chanlocs", []) if isinstance(chan, dict)]


def _common_channel_labels(datasets: list[dict[str, Any]]) -> list[str]:
    first = _channel_labels(datasets[0])
    other_sets = [set(label.lower() for label in _channel_labels(dataset)) for dataset in datasets[1:]]
    return [label for label in first if all(label.lower() in labels for labels in other_sets)]


__all__ = ["pop_fusechanrej"]

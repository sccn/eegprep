"""Create STUDY trialinfo rows from loaded EEG datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import as_alleeg_list, ensure_study, sync_datasetinfo, trialinfo_from_eeg


EVENT_TRIALINFO_EXCLUDE = {"latency", "urevent", "epoch"}


def std_maketrialinfo(
    STUDY: dict[str, Any] | None,
    ALLEEG: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """Populate ``STUDY.datasetinfo[*].trialinfo`` from loaded EEG metadata."""
    datasets = as_alleeg_list(ALLEEG)
    study = sync_datasetinfo(ensure_study(STUDY), datasets)
    alltrialinfo: list[list[dict[str, Any]]] = []
    for index, eeg in enumerate(datasets):
        rows = trialinfo_from_eeg(eeg)
        if not rows:
            rows = _trialinfo_from_events(eeg)
        alltrialinfo.append(rows)
        if rows and index < len(study.get("datasetinfo") or []):
            study["datasetinfo"][index]["trialinfo"] = rows
    return study, alltrialinfo


def _trialinfo_from_events(eeg: dict[str, Any]) -> list[dict[str, Any]]:
    trials = int(eeg.get("trials", 1) or 1)
    if trials <= 1:
        return []
    events = _event_rows(eeg.get("event"))
    by_epoch: dict[int, dict[str, Any]] = {}
    for event in events:
        epoch = _event_epoch(event)
        if epoch is None or epoch in by_epoch:
            continue
        row = {
            key: deepcopy(value)
            for key, value in event.items()
            if key not in EVENT_TRIALINFO_EXCLUDE and not _empty_value(value) and not isinstance(value, dict)
        }
        if row:
            by_epoch[epoch] = row
    if not by_epoch:
        return []
    return [by_epoch.get(epoch, {}) for epoch in range(1, trials + 1)]


def _event_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [event for event in value if isinstance(event, dict)]


def _event_epoch(event: dict[str, Any]) -> int | None:
    value = event.get("epoch")
    if _empty_value(value):
        return None
    if isinstance(value, np.ndarray):
        value = value.ravel()[0]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if _empty_value(value):
        return None
    epoch = int(value)
    return epoch if epoch >= 1 else None


def _empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple, dict, set)) and len(value) == 0


__all__ = ["std_maketrialinfo"]

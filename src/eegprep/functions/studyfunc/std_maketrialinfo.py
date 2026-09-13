"""Create STUDY trialinfo rows from loaded EEG datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.studyfunc._study_utils import sync_study_datasets, trialinfo_from_eeg


EVENT_TRIALINFO_EXCLUDE = {"latency", "urevent", "epoch"}


def std_maketrialinfo(
    STUDY: dict[str, Any] | None,
    ALLEEG: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """Populate trial information from each epoch's time-locking event."""
    study, datasets = sync_study_datasets(STUDY, ALLEEG)
    alltrialinfo: list[list[dict[str, Any]]] = []
    for index, eeg in enumerate(datasets):
        rows = _trialinfo_from_events(eeg)
        if not rows:
            rows = trialinfo_from_eeg(eeg)
        alltrialinfo.append(rows)
        if rows and index < len(study.get("datasetinfo") or []):
            study["datasetinfo"][index]["trialinfo"] = rows
    return study, alltrialinfo


def _trialinfo_from_events(eeg: dict[str, Any]) -> list[dict[str, Any]]:
    trials = int(eeg.get("trials", 1) or 1)
    if trials <= 1:
        return []
    events = _event_rows(eeg.get("event"))
    by_epoch: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        epoch = _event_epoch(event)
        if epoch is None:
            continue
        by_epoch.setdefault(epoch, []).append(event)
    if set(by_epoch) != set(range(1, trials + 1)):
        return []
    rows = []
    for epoch in range(1, trials + 1):
        event = min(by_epoch[epoch], key=lambda item: _time_lock_distance(item, epoch, eeg))
        rows.append(
            {
                key: deepcopy(value)
                for key, value in event.items()
                if key not in EVENT_TRIALINFO_EXCLUDE and not _empty_value(value) and not isinstance(value, dict)
            }
        )
    return rows if any(rows) else []


def _time_lock_distance(event: dict[str, Any], epoch: int, eeg: dict[str, Any]) -> float:
    latency = float(event.get("latency", np.inf))
    pnts = int(eeg.get("pnts", 0) or 0)
    srate = float(eeg.get("srate", 1.0) or 1.0)
    xmin = float(eeg.get("xmin", 0.0) or 0.0)
    zero_latency = (epoch - 1) * pnts - xmin * srate + 1.0
    return abs(latency - zero_latency)


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

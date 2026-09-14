"""Compute target-event delays from preceding original events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def eeg_time2prev(
    EEG: dict[str, Any],
    target: Sequence[Any],
    previous: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return delays from target events to preceding original events.

    Event and urevent pointers in EEGPrep dictionaries are zero-based. Returned
    ``targets``, ``urtargets``, and ``urprevs`` therefore use zero-based indices;
    ``urprevs`` uses ``-1`` when no matching preceding event exists. Event
    latencies remain EEGLAB-compatible one-based sample positions.

    Args:
        EEG: Dataset containing ``event``, ``urevent``, and ``srate``.
        target: Event types whose delays should be reported.
        previous: Event types eligible as the preceding event.

    Returns:
        ``(delays_ms, targets, urtargets, urprevs)``.
    """
    target_types = _type_names(target, "target")
    previous_types = _type_names(previous, "previous")
    events = _records(EEG.get("event"), "event")
    urevents = _records(EEG.get("urevent"), "urevent")
    sampling_rate = float(EEG.get("srate", 0))
    if sampling_rate <= 0:
        raise ValueError("eeg_time2prev: EEG.srate must be positive")

    delays: list[float] = []
    targets: list[int] = []
    urtargets: list[int] = []
    urprevs: list[int] = []
    for event_index, event in enumerate(events):
        urevent_index = _urevent_index(event, len(urevents))
        if _event_type(urevents[urevent_index].get("type")) not in target_types:
            continue

        previous_index = next(
            (
                index
                for index in range(urevent_index - 1, -1, -1)
                if _event_type(urevents[index].get("type")) in previous_types
            ),
            -1,
        )
        delay = 0.0
        if previous_index >= 0:
            target_latency = float(urevents[urevent_index]["latency"])
            previous_latency = float(urevents[previous_index]["latency"])
            delay = (target_latency - previous_latency) * 1000 / sampling_rate

        delays.append(delay)
        targets.append(event_index)
        urtargets.append(urevent_index)
        urprevs.append(previous_index)

    return (
        np.asarray(delays, dtype=float),
        np.asarray(targets, dtype=int),
        np.asarray(urtargets, dtype=int),
        np.asarray(urprevs, dtype=int),
    )


def _type_names(values: Sequence[Any], name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f'eeg_time2prev: {name} must be a sequence of event types')
    return {_event_type(value) for value in values}


def _event_type(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).casefold()


def _records(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        raise ValueError(f"eeg_time2prev: EEG.{name} is required")
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise ValueError(f"eeg_time2prev: EEG.{name} must contain event dictionaries")
    return value


def _urevent_index(event: dict[str, Any], count: int) -> int:
    if "urevent" not in event:
        raise ValueError("eeg_time2prev: every EEG.event needs a zero-based urevent pointer")
    index = int(event["urevent"])
    if index != event["urevent"] or index < 0 or index >= count:
        raise ValueError("eeg_time2prev: EEG.event contains an invalid zero-based urevent pointer")
    return index


__all__ = ["eeg_time2prev"]

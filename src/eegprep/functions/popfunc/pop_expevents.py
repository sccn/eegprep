"""Export EEG events to a text file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, is_empty_value


def pop_expevents(EEG: dict[str, Any], filename: str | Path, unit: str = "samples") -> str:
    """Export ``EEG.event`` as a tab-delimited text file.

    Columns follow ``eeg_eventtable.m`` lines 123-159: a 1-based ``number``
    column, then event fields in first-seen order. Numeric cells use ``%f``.
    The ``urevent`` column is written 1-based. ``unit`` is ``samples`` (the
    GUI and function default) or ``seconds``.
    """
    if unit not in {"samples", "seconds"}:
        raise ValueError("pop_expevents: unit must be 'samples' or 'seconds'. Pass unit='samples' or unit='seconds'.")
    events = events_to_records(EEG.get("event"))
    srate = float(EEG.get("srate") or 1.0)
    for event in events:
        if not is_empty_value(event.get("urevent")):
            event["urevent"] = int(event["urevent"]) + 1
        if unit == "seconds":
            for key in ("latency", "duration"):
                if key in event and _is_number(event[key]):
                    event[key] = float(event[key]) / srate
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _field_order(events)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["number", *fields])
        for index, event in enumerate(events, start=1):
            writer.writerow([_format_cell(index), *[_format_cell(event.get(field, "")) for field in fields]])
    return f"LASTCOM = pop_expevents(EEG, {format_history_value(path)}, {format_history_value(unit)});"


def _field_order(events: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for event in events:
        for key in event:
            if key in seen:
                continue
            seen.add(key)
            fields.append(key)
    return fields


def _format_cell(value: Any) -> str:
    if _is_number(value):
        return f"{float(value):.6f}"
    return "" if value is None else str(value)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, str):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value))
    return False

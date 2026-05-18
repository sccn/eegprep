"""Export EEG events to a text file."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_expevents(EEG: dict[str, Any], filename: str | Path) -> str:
    """Export ``EEG.event`` as a tab-delimited text file."""
    events = events_to_records(EEG.get("event"))
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for event in events for key in event}) if events else ["type", "latency"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    return f"LASTCOM = pop_expevents(EEG, {format_history_value(path)});"

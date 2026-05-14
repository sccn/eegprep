"""Export EEGPrep datasets to a minimal BIDS EEG folder."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import _channel_labels, events_to_records, write_json
from eegprep.functions.popfunc.pop_saveset import pop_saveset


def pop_exportbids(
    EEG: dict[str, Any] | list[dict[str, Any]],
    output_dir: str | Path,
    *,
    subject: str = "01",
    task: str = "eeg",
    return_com: bool = False,
) -> str | tuple[str, str]:
    """Export EEG dataset(s) as BIDS-like EEGLAB files and sidecars."""
    root = Path(output_dir)
    datasets = EEG if isinstance(EEG, list) else [EEG]
    root.mkdir(parents=True, exist_ok=True)
    write_json(
        root / "dataset_description.json",
        {"Name": "EEGPrep export", "BIDSVersion": "1.9.0", "DatasetType": "raw"},
    )
    _write_participants(root / "participants.tsv", subject)
    for run_index, eeg in enumerate(datasets, start=1):
        eeg_dir = root / f"sub-{subject}" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"sub-{subject}_task-{task}_run-{run_index:02d}"
        pop_saveset(deepcopy(eeg), str(eeg_dir / f"{prefix}_eeg.set"))
        _write_channels(eeg_dir / f"{prefix}_channels.tsv", eeg)
        _write_events(eeg_dir / f"{prefix}_events.tsv", eeg)
    command = f"LASTCOM = pop_exportbids(EEG, '{root}');"
    return (str(root), command) if return_com else str(root)


def _write_participants(path: Path, subject: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["participant_id"])
        writer.writerow([f"sub-{subject}"])


def _write_channels(path: Path, eeg: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["name", "type", "units"])
        for label in _channel_labels(eeg):
            writer.writerow([label, "EEG", "uV"])


def _write_events(path: Path, eeg: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["onset", "duration", "trial_type"])
        srate = float(eeg.get("srate", 1) or 1)
        for event in events_to_records(eeg.get("event")):
            onset = (float(event.get("latency", 1)) - 1) / srate
            duration = float(event.get("duration", 0) or 0) / srate
            writer.writerow([f"{onset:.10g}", f"{duration:.10g}", event.get("type", "event")])

"""Small BIDS tool helpers for EEGPrep GUI File menu actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from eegprep.functions.popfunc._file_io import write_json
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.plugins.EEG_BIDS.bids_list_eeg_files import bids_list_eeg_files
from eegprep.plugins.EEG_BIDS.pop_exportbids import pop_exportbids


def bids_exporter(EEG: dict[str, Any] | list[dict[str, Any]], output_dir: str | Path) -> tuple[str, str]:
    """Run the EEGPrep BIDS export wizard backend."""
    return pop_exportbids(EEG, output_dir, return_com=True)


def pop_taskinfo(target: dict[str, Any], **metadata: Any) -> tuple[dict[str, Any], str]:
    """Attach BIDS task metadata to an EEG or STUDY dict."""
    return _update_bids_metadata(target, "task", metadata, "pop_taskinfo")


def pop_participantinfo(target: dict[str, Any], **metadata: Any) -> tuple[dict[str, Any], str]:
    """Attach BIDS participant metadata to an EEG or STUDY dict."""
    return _update_bids_metadata(target, "participant", metadata, "pop_participantinfo")


def pop_eventinfo(target: dict[str, Any], **metadata: Any) -> tuple[dict[str, Any], str]:
    """Attach BIDS event metadata to an EEG or STUDY dict."""
    return _update_bids_metadata(target, "event", metadata, "pop_eventinfo")


def validate_bids(root: str | Path) -> dict[str, list[str]]:
    """Return a lightweight validation report for a BIDS EEG folder."""
    path = Path(root)
    errors = []
    warnings = []
    if not path.exists():
        errors.append(f"BIDS path does not exist: {path}")
    if not (path / "dataset_description.json").exists():
        warnings.append("dataset_description.json is missing")
    try:
        files = bids_list_eeg_files(str(path))
    except Exception as exc:
        errors.append(str(exc))
        files = []
    if not files:
        warnings.append("No supported EEG files were found")
    report = {"errors": errors, "warnings": warnings}
    write_json(path / "eegprep_bids_validation.json", report) if path.exists() else None
    return report


def _update_bids_metadata(
    target: dict[str, Any],
    section: str,
    metadata: dict[str, Any],
    command_name: str,
) -> tuple[dict[str, Any], str]:
    etc = target.setdefault("etc", {})
    bids = etc.setdefault("bids", {})
    bids.setdefault(section, {}).update(metadata)
    target_name = "STUDY" if "datasetinfo" in target else "EEG"
    pieces = [target_name]
    for key, value in metadata.items():
        pieces.extend([format_history_value(key), format_history_value(value)])
    return target, f"LASTCOM = {command_name}({', '.join(pieces)});"

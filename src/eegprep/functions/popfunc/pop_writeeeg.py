"""Export EEG data to EDF/BDF/GDF through MNE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mne.export import export_raw

from eegprep.functions.popfunc._file_io import eeg_to_mne_raw


def pop_writeeeg(EEG: dict[str, Any], filename: str | Path, *args: Any, **kwargs: Any) -> str:
    """Write continuous EEG data to an EDF/BDF/GDF-compatible file."""
    path = Path(filename)
    if path.suffix.lower() not in {".edf", ".bdf", ".gdf"}:
        raise ValueError("pop_writeeeg output must end in .edf, .bdf, or .gdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = eeg_to_mne_raw(EEG)
    export_raw(str(path), raw, fmt=path.suffix.lower()[1:], overwrite=True)
    return f"LASTCOM = pop_writeeeg(EEG, '{path}');"

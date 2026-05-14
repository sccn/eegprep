"""Import EEG data using Python file readers analogous to EEGLAB File-IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne

from eegprep.functions.popfunc._file_io import mne_raw_to_eeg
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_loadset import pop_loadset


def pop_fileio(filename: str | Path, *, return_com: bool = False, **kwargs: Any) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a supported EEG data file with MNE/File-IO-style readers."""
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == ".set":
        eeg = pop_loadset(str(path))
    elif suffix == ".mat" and kwargs.get("dataformat") != "matlab-array":
        try:
            eeg = pop_loadset(str(path))
        except Exception:
            eeg = pop_importdata("data", str(path), "setname", path.stem, "dataformat", "matlab", **kwargs)
    elif suffix in {".csv", ".txt", ".tsv", ".npy", ".npz"}:
        eeg = pop_importdata("data", str(path), "setname", path.stem, **kwargs)
    else:
        reader = _reader_for_suffix(suffix)
        raw = reader(str(path), preload=True, verbose=False)
        eeg = mne_raw_to_eeg(raw, setname=path.stem, filename=str(path))
    command = f"EEG = pop_fileio({format_history_value(path)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _reader_for_suffix(suffix: str):
    if suffix in {".edf", ".bdf"}:
        return mne.io.read_raw_edf
    if suffix == ".gdf":
        return mne.io.read_raw_gdf
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision
    if suffix == ".mff":
        return mne.io.read_raw_egi
    if suffix == ".cnt":
        return mne.io.read_raw_cnt
    if suffix == ".eeg":
        return mne.io.read_raw_brainvision
    raise ValueError(f"Unsupported File-IO import format: {suffix or '<none>'}")

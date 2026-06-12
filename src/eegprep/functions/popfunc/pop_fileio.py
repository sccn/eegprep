"""Import EEG data using Python file readers analogous to EEGLAB File-IO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import scipy.io

from eegprep.functions.popfunc._file_io import mne_raw_to_eeg
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_loadset import _is_hdf5_file, pop_loadset

logger = logging.getLogger(__name__)

# Fields that mark a .mat as a saved EEGLAB dataset rather than a raw data array.
_EEGLAB_STRUCT_MARKERS = frozenset({"nbchan", "srate", "pnts", "trials", "chanlocs", "setname", "xmin", "xmax"})


def pop_fileio(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a supported EEG data file with MNE/File-IO-style readers."""
    path = Path(filename)
    suffix = path.suffix.lower()
    if suffix == ".set":
        eeg = pop_loadset(str(path))
    elif suffix == ".mat" and kwargs.get("dataformat") != "matlab-array":
        if _mat_is_eeglab_dataset(path):
            logger.info("pop_fileio: loading %s as an EEGLAB dataset", path)
            eeg = pop_loadset(str(path))
        else:
            logger.info("pop_fileio: importing %s as a raw MATLAB data array", path)
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


def _mat_is_eeglab_dataset(path: Path) -> bool:
    """Return True when a .mat file holds an EEGLAB dataset rather than a raw data array.

    A dataset is recognized by an ``EEG`` struct variable or by top-level EEGLAB marker
    fields (a .set saved with ``-struct``). MAT v7.3 files are HDF5 and always EEGLAB sets.
    """
    if _is_hdf5_file(path):
        return True
    names = {name for name, _shape, _cls in scipy.io.whosmat(str(path))}
    return "EEG" in names or bool(names & _EEGLAB_STRUCT_MARKERS)


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

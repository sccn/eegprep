"""Import EEG data using Python file readers analogous to EEGLAB File-IO."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.io

from eegprep.functions.popfunc._file_io import mne_raw_to_eeg
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_numeric_sequence
from eegprep.functions.popfunc.pop_importdata import pop_importdata
from eegprep.functions.popfunc.pop_loadcnt import pop_loadcnt
from eegprep.functions.popfunc.pop_loadset import _is_hdf5_file, pop_loadset
from eegprep.functions.popfunc.pop_select import pop_select

logger = logging.getLogger(__name__)

# Fields that mark a .mat as a saved EEGLAB dataset rather than a raw data array.
_EEGLAB_STRUCT_MARKERS = frozenset({"nbchan", "srate", "pnts", "trials", "chanlocs", "setname", "xmin", "xmax"})


def pop_fileio(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import an EEG file, optionally selecting 1-based channels, samples, or trials."""
    path = Path(filename)
    suffix = path.suffix.lower()
    history_options = dict(kwargs)
    blockrange = kwargs.pop("blockrange", None)
    channels = kwargs.pop("channels", None)
    samples = kwargs.pop("samples", None)
    trials = kwargs.pop("trials", None)
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
    elif suffix == ".cnt":
        if blockrange is not None:
            start, stop = _blockrange_values(blockrange)
            kwargs["t1"] = start
            kwargs["lddur"] = stop - start
        eeg = pop_loadcnt(path, **kwargs)
    else:
        reader = _reader_for_suffix(suffix)
        raw = reader(str(path), preload=True, verbose=False)
        if blockrange is not None:
            _crop_raw_to_blockrange(raw, blockrange)
        eeg = mne_raw_to_eeg(raw, setname=path.stem, filename=str(path))
    eeg = _select_imported_data(eeg, channels=channels, samples=samples, trials=trials)
    command = _history_command(path, history_options)
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
    if suffix == ".edf":
        return mne.io.read_raw_edf
    if suffix == ".bdf":
        return mne.io.read_raw_bdf
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


def _crop_raw_to_blockrange(raw: mne.io.BaseRaw, blockrange: Any) -> None:
    start, stop = _blockrange_values(blockrange)
    recording_stop = raw.n_times / float(raw.info["sfreq"])
    if start >= recording_stop:
        raise ValueError("blockrange starts after the end of the recording")
    raw.crop(tmin=start, tmax=min(stop, recording_stop), include_tmax=False)


def _blockrange_values(blockrange: Any) -> tuple[float, float]:
    values = np.asarray(blockrange, dtype=float).reshape(-1)
    if values.size != 2 or not np.all(np.isfinite(values)):
        raise ValueError("blockrange must contain two finite times in seconds")
    start, stop = (float(value) for value in values)
    if start < 0 or stop <= start:
        raise ValueError("blockrange must satisfy 0 <= start < stop")
    return start, stop


def _select_imported_data(EEG: dict[str, Any], *, channels: Any, samples: Any, trials: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if channels is not None:
        selected = parse_numeric_sequence(channels, dtype=int)
        if not selected or any(index < 1 for index in selected):
            raise ValueError("channels must contain positive 1-based indices")
        options["channel"] = [index - 1 for index in selected]
    if samples is not None:
        options["point"] = _inclusive_bounds(samples, "samples")
    if trials is not None:
        start, stop = _inclusive_bounds(trials, "trials")
        options["trial"] = list(range(start, stop + 1))
    if not options:
        return EEG
    return pop_select(EEG, gui=False, **options)


def _inclusive_bounds(value: Any, name: str) -> list[int]:
    bounds = parse_numeric_sequence(value, dtype=int)
    if len(bounds) != 2 or bounds[0] < 1 or bounds[1] < bounds[0]:
        raise ValueError(f"{name} must be a positive 1-based [start, stop] range")
    return bounds


def _history_command(path: Path, options: dict[str, Any]) -> str:
    pieces = [format_history_value(path)]
    for key, value in options.items():
        pieces.extend([format_history_value(key), format_history_value(value, cell_for_sequence=None)])
    return f"EEG = pop_fileio({', '.join(pieces)});"

"""Raw EEG file readers used by EEG-BIDS import workflows."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_loadbv import pop_loadbv


def load_raw_eeg_file(
    filename: str,
    *,
    dtype: np.dtype,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    """Load a supported BIDS raw EEG file into an EEG dictionary."""
    _path, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext == ".set":
        from eegprep.functions.popfunc.pop_loadset import pop_loadset

        eeg = pop_loadset(filename)
        eeg["data"] = eeg["data"].astype(dtype)
        report = {"ImporterUsed": "pop_loadset"}
        srate = eeg["srate"]
        times_sec = eeg["times"] / 1000.0
        return eeg, srate, times_sec, report

    if ext == ".vhdr":
        eeg, srate, times_sec, raw_report = _load_brainvision_file(filename, dtype=dtype)
        return eeg, srate, times_sec, raw_report

    if ext in {".edf", ".bdf"}:
        eeg, srate, times_sec, raw_report = _load_biosig_file(filename, dtype=dtype)
        return eeg, srate, times_sec, raw_report

    if ext in [".fdt", ".vmrk", ".eeg"]:
        raise ValueError(
            f"pop_load_frombids should be called with the main data file, but was called on a sidecar file: {filename}."
        )
    raise ValueError(f"Unsupported file format: {ext}. Supported formats are .set, .edf, .bdf, .vhdr.")


def _load_brainvision_file(
    filename: str,
    *,
    dtype: np.dtype,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    eeg = pop_loadbv(filename)
    eeg["data"] = np.asarray(eeg["data"], dtype=dtype)
    srate = float(np.asarray(eeg["srate"]).reshape(-1)[0])
    times_sec = np.asarray(eeg["times"], dtype=float) / 1000.0
    eeg["chaninfo"].setdefault("nosedir", "+X")
    return eeg, srate, times_sec, {"ImporterUsed": "pop_loadbv"}


def _load_biosig_file(
    filename: str,
    *,
    dtype: np.dtype,
) -> tuple[dict[str, Any], float, np.ndarray, dict[str, Any]]:
    """Read EDF or BDF with ``pop_biosig``, matching EEGLAB ``pop_importbids``."""
    eeg = pop_biosig(filename)
    eeg["data"] = np.asarray(eeg["data"], dtype=dtype)
    srate = float(np.asarray(eeg["srate"]).reshape(-1)[0])
    times_sec = np.asarray(eeg["times"], dtype=float) / 1000.0
    eeg["chaninfo"].setdefault("nosedir", "+X")
    return eeg, srate, times_sec, {"ImporterUsed": "pop_biosig"}

"""Import simple BCI2000 ASCII or MATLAB signal files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc._pop_utils import format_history_value


def pop_loadbci(
    filename: str | Path,
    srate: float = 256.0,
    *,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a BCI2000-style file into an EEG dictionary."""
    path = Path(filename)
    eeg = _load_mat(path, srate) if _looks_like_mat_file(path) else _load_ascii(path, srate)
    command = f"EEG = pop_loadbci({format_history_value(path)}, {float(srate):.12g});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _load_ascii(path: Path, srate: float) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        labels = stream.readline().split()
    if not labels or len(labels) > 300:
        raise ValueError("Not a BCI ASCII file")
    data = np.loadtxt(path, skiprows=1, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] == len(labels):
        data = data.T
    chanlocs = [{"labels": label} for label in labels]
    return eeg_from_data(
        data,
        srate=float(srate),
        setname=path.stem,
        nbchan=len(labels),
        chanlocs=chanlocs,
        filename=path.name,
        filepath=str(path.parent),
    )


def _load_mat(path: Path, srate: float) -> dict[str, Any]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "signal" not in mat:
        raise ValueError("BCI MATLAB files must contain a 'signal' variable")
    signal = np.asarray(mat["signal"], dtype=float)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    labels = [f"C{index}" for index in range(1, signal.shape[1] + 1)]
    extras = []
    for key, value in mat.items():
        if key.startswith("__") or key == "signal":
            continue
        array = np.asarray(value, dtype=float).reshape(-1)
        if array.shape[0] == signal.shape[0]:
            extras.append(array)
            labels.append(key)
    data = np.column_stack([signal, *extras]).T if extras else signal.T
    chanlocs = [{"labels": label} for label in labels]
    return eeg_from_data(
        data,
        srate=float(srate),
        setname=path.stem,
        nbchan=len(labels),
        chanlocs=chanlocs,
        filename=path.name,
        filepath=str(path.parent),
    )


def _looks_like_mat_file(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(6) == b"MATLAB"


__all__ = ["pop_loadbci"]

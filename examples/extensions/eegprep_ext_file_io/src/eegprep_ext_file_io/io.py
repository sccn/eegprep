"""CSV import/export example functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pop_demo_import_csv(filename: str | Path, *, srate: float = 1.0, return_com: bool = False):
    """Import a channel-major CSV file into a minimal EEG dictionary."""
    path = Path(filename)
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    eeg = {
        "setname": path.stem,
        "filename": path.name,
        "filepath": str(path.parent),
        "data": data,
        "nbchan": int(data.shape[0]),
        "pnts": int(data.shape[1]),
        "trials": 1,
        "srate": float(srate),
        "xmin": 0.0,
        "xmax": (data.shape[1] - 1) / float(srate),
        "times": np.arange(data.shape[1]) / float(srate),
        "chanlocs": [{"labels": f"Ch{index + 1}"} for index in range(data.shape[0])],
        "event": [],
        "urevent": [],
        "epoch": [],
        "history": "",
        "etc": {"eegprep_ext_file_io": {"source": "csv"}},
    }
    command = f"EEG = pop_demo_import_csv({str(path)!r}, srate={float(srate):g});"
    return (eeg, command) if return_com else eeg


def pop_demo_export_csv(
    EEG: dict[str, Any],
    filename: str | Path,
    *,
    return_com: bool = False,
):
    """Export EEG.data to a channel-major CSV file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(EEG["data"], dtype=float), delimiter=",")
    command = f"LASTCOM = pop_demo_export_csv(EEG, {str(path)!r});"
    return (path, command) if return_com else path

"""Import SnapMaster files into EEGPrep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.sigprocfunc.snapread import snapread


def pop_snapread(
    filename: str | Path,
    gain: float = 1.0,
    *,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a SnapMaster ``.SMA`` file."""
    data, params, events, _header = snapread(filename)
    data = data * float(gain)
    path = Path(filename)
    eeg = eeg_from_data(
        data,
        srate=float(params[2]),
        setname="SnapMaster file",
        comments=f"Original file: {path}",
        chanlocs=[{"labels": f"Ch{index}"} for index in range(1, data.shape[0] + 1)],
        filename=path.name,
        filepath=str(path.parent),
    )
    event_indices = np.flatnonzero(events)
    if event_indices.size:
        eeg["event"] = [{"type": int(events[index]), "latency": float(index + 1)} for index in event_indices]
    eeg = eeg_checkset(eeg, "eventconsistency")
    command = f"EEG = pop_snapread({format_history_value(path)}, {float(gain):.12g});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


__all__ = ["pop_snapread"]

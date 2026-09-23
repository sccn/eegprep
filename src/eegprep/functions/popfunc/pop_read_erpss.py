"""Import ERPSS ``.RAW`` and ``.RDF`` recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.sigprocfunc.read_erpss import read_erpss


def pop_read_erpss(
    filename: str | Path,
    srate: float | None = None,
    *,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import an ERPSS recording into an EEG dictionary.

    A valid sampling rate stored in the recording takes precedence over
    ``srate``. Pass ``srate`` for older recordings whose headers do not carry
    enough timing information.
    """
    path = Path(filename)
    data, raw_events, header = read_erpss(path)
    effective_srate = _effective_srate(header["srate"], srate)
    eeg = eeg_from_data(
        data,
        srate=effective_srate,
        setname="ERPSS data",
        comments=f"Original file: {path}",
        chanlocs=[{"labels": label} for label in header["chanlabels"]],
        filename=path.name,
        filepath=str(path.parent),
    )
    events = []
    for index, event in enumerate(raw_events):
        events.append(
            {
                **event,
                "type": event["event_code"],
                "latency": float(event["sample_offset"]),
                "urevent": index,
            }
        )
    eeg["event"] = events
    eeg["urevent"] = np.asarray(
        [{key: value for key, value in event.items() if key != "urevent"} for event in events],
        dtype=object,
    )
    eeg = eeg_checkset(eeg, "eventconsistency")
    command = f"EEG = pop_read_erpss({format_history_value(path)}, {format_history_value(effective_srate)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _effective_srate(header_srate: Any, fallback: float | None) -> float:
    file_srate = float(header_srate)
    if np.isfinite(file_srate) and file_srate >= 0.5:
        return file_srate
    if fallback is None:
        raise ValueError("ERPSS header has no valid sampling rate; pass srate explicitly")
    fallback = float(fallback)
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError("srate must be a finite positive number")
    return fallback


__all__ = ["pop_read_erpss"]

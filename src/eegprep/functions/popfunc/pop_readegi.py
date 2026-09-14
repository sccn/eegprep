"""Import an EGI Simple Binary RAW file into an EEG dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.sigprocfunc.readegi import readegi
from eegprep.functions.sigprocfunc.readegilocs import readegilocs


def pop_readegi(
    filename: str | Path,
    datachunks: int | Sequence[int] | np.ndarray | None = None,
    forceversion: int | None = None,
    fileloc: str | Path | None = "auto",
    *,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import an EGI Simple Binary file.

    ``datachunks`` uses EEGLAB-facing 1-based frame or segment numbers. EGI
    integer A/D values are converted to microvolts by :func:`readegi`, event
    channels become dataset events, and segmented recordings are returned as
    channel-by-point-by-trial arrays.
    """
    path = Path(filename)
    header, trial_data, event_data, categories = readegi(path, datachunks, forceversion)
    command = _history_command(path, datachunks, forceversion, fileloc)
    eeg = _egi_to_eeg(
        header,
        trial_data,
        event_data,
        categories,
        filename=path,
        fileloc=fileloc,
        comments=f"Original file: {path}",
    )
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _egi_to_eeg(
    header: dict[str, Any],
    trial_data: np.ndarray,
    event_data: np.ndarray,
    categories: np.ndarray,
    *,
    filename: Path,
    fileloc: str | Path | None,
    comments: str,
) -> dict[str, Any]:
    signal_channels = int(header["nchan"])
    channel_locations = [{"labels": f"E{index}"} for index in range(1, signal_channels + 1)]
    eeg = eeg_from_data(
        trial_data,
        nbchan=signal_channels,
        srate=float(header["samp_rate"]),
        setname="EGI file",
        comments=comments,
        filename=filename.name,
        filepath=str(filename.parent),
        chanlocs=channel_locations,
    )

    points, trials = _dataset_dimensions(header, trial_data.shape[1])
    eeg["pnts"] = points
    eeg["trials"] = trials
    eeg["xmax"] = (points - 1) / float(header["samp_rate"]) if points else 0.0
    eeg["times"] = np.arange(points, dtype=float) / float(header["samp_rate"]) * 1000.0
    eeg["event"] = np.asarray(_event_records(event_data, header["eventcode"], points, trials), dtype=object)

    if _has_empty_reference(eeg["data"]):
        eeg["data"] = np.asarray(eeg["data"][:-1, :])
        eeg["nbchan"] = int(eeg["data"].shape[0])
        eeg["chanlocs"] = np.asarray(eeg["chanlocs"][:-1], dtype=object)

    if trials > 1:
        flat = np.asarray(eeg["data"])
        eeg["data"] = flat.reshape(flat.shape[0], trials, points).transpose(0, 2, 1)
    _apply_segment_categories(eeg, header, categories)
    _rebuild_urevents(eeg)
    with strict_mode(False):
        eeg = eeg_checkset(eeg, "eventconsistency")
    if fileloc:
        eeg = readegilocs(eeg, None if str(fileloc).lower() == "auto" else str(fileloc))
    eeg["saved"] = "no"
    return eeg


def _event_records(
    event_data: np.ndarray,
    event_codes: list[str],
    points: int,
    trials: int,
) -> list[dict[str, Any]]:
    if event_data.shape[1] == 0:
        return []
    events = []
    for event_index in range(event_data.shape[0] - 1, -1, -1):
        values = np.asarray(event_data[event_index])
        differences = np.diff(np.abs(np.r_[0, values]))
        for latency in np.flatnonzero(differences > 0) + 1:
            event = {"type": event_codes[event_index], "latency": int(latency)}
            if trials > 1:
                event["epoch"] = 1 + int((latency - 1) // points)
            events.append(event)
    events.sort(key=lambda event: (int(event.get("epoch", 0)), int(event["latency"])))
    return events


def _dataset_dimensions(header: dict[str, Any], loaded_samples: int) -> tuple[int, int]:
    if not header["segmented"]:
        return loaded_samples, 1
    points = int(header["segsamps"])
    if points <= 0 or loaded_samples % points:
        raise ValueError("Segmented EGI data do not contain complete equal-length trials")
    return points, loaded_samples // points


def _has_empty_reference(data: np.ndarray) -> bool:
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        return False
    return bool(np.all(values[-1] == 0))


def _apply_segment_categories(
    eeg: dict[str, Any],
    header: dict[str, Any],
    categories: np.ndarray,
) -> None:
    if eeg["trials"] <= 1 or categories.size == 0:
        return
    events = [dict(event) for event in eeg.get("event", [])]
    if not events:
        events = [
            {
                "epoch": trial + 1,
                "type": "TLE",
                "latency": 1 + trial * int(eeg["pnts"]),
            }
            for trial in range(int(eeg["trials"]))
        ]
    else:
        points = int(eeg["pnts"])
        for event in events:
            event.setdefault("epoch", 1 + int((float(event["latency"]) - 1) // points))
    names = list(header["catname"])
    for event in events:
        epoch = int(event.get("epoch", 0))
        if not 1 <= epoch <= categories.size:
            continue
        category_index = int(categories[epoch - 1])
        if 1 <= category_index <= len(names):
            event["category"] = names[category_index - 1]
    eeg["event"] = np.asarray(events, dtype=object)


def _rebuild_urevents(eeg: dict[str, Any]) -> None:
    events = [dict(event) for event in eeg.get("event", [])]
    urevents = []
    for index, event in enumerate(events):
        urevent = dict(event)
        urevent.pop("urevent", None)
        event["urevent"] = index
        urevents.append(urevent)
    eeg["event"] = np.asarray(events, dtype=object)
    eeg["urevent"] = np.asarray(urevents, dtype=object)


def _history_command(
    filename: Path,
    datachunks: int | Sequence[int] | np.ndarray | None,
    forceversion: int | None,
    fileloc: str | Path | None,
) -> str:
    arguments = [format_history_value(filename)]
    if datachunks is not None or forceversion is not None or fileloc != "auto":
        arguments.append(format_history_value([] if datachunks is None else datachunks))
    if forceversion is not None or fileloc != "auto":
        arguments.append(format_history_value(forceversion, none_as_empty=True))
    if fileloc != "auto":
        arguments.append(format_history_value(fileloc, none_as_empty=True))
    return f"EEG = pop_readegi({', '.join(arguments)});"


__all__ = ["pop_readegi"]

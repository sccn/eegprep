"""Import EEG data using BIOSIG-equivalent Python readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.sigprocfunc.openbdf import openbdf
from eegprep.functions.sigprocfunc.readbdf import readbdf


_BIOSIG_SUFFIXES = {".edf", ".bdf", ".gdf"}
_STATUS_LABEL = "Status"
_BIOSEMI_MAGIC = b"\xffBIOSEMI"
_TRIGGER_MASK = 0xFFFF
_EPOCH_BIT = 0x10000


def pop_biosig(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import BIOSIG-style EDF/BDF/GDF files, optionally over a time range.

    BioSemi BDF files drop the ``Status`` channel after trigger events are
    read from its rising edges.
    """
    path = Path(filename)
    if path.suffix.lower() not in _BIOSIG_SUFFIXES:
        raise ValueError(
            "pop_biosig supports EDF, BDF, and GDF files. Use pop_fileio or pop_loadset for other formats."
        )
    blockrange = kwargs.get("blockrange")
    eeg, _command = pop_fileio(filename, return_com=True, **kwargs)
    if path.suffix.lower() == ".bdf":
        eeg = _import_bdf_status_channel(eeg, path, blockrange)
    arguments = [format_history_value(path)]
    for key, value in kwargs.items():
        arguments.extend([format_history_value(key), format_history_value(value)])
    command = f"EEG = pop_biosig({', '.join(arguments)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _import_bdf_status_channel(eeg: dict[str, Any], path: Path, blockrange: Any) -> dict[str, Any]:
    labels = [_channel_label(loc) for loc in list(eeg.get("chanlocs", []))]
    if _STATUS_LABEL not in labels or not _is_biosemi(path):
        return eeg
    opened = openbdf(path)
    head = opened["Head"]
    file_labels = [str(label).strip() for label in head["Label"]]
    status_index = file_labels.index(_STATUS_LABEL)
    record_count = int(head["NRec"])
    samples_per_record = int(head["SPR"][status_index])
    total_samples = record_count * samples_per_record
    if blockrange is None:
        start_sample = 0
        stop_sample = total_samples
    else:
        start, stop = _blockrange_seconds(blockrange)
        sample_rate = float(head["SampleRate"][status_index])
        start_sample = int(start * sample_rate)
        stop_sample = int(stop * sample_rate)
    decoded, _raw = readbdf(opened, np.arange(1, record_count + 1), 3)
    status = np.asarray(decoded["Record"][status_index], dtype=np.int64)
    _replace_events(eeg, _status_events(status, start_sample, stop_sample))
    _drop_channel(eeg, labels.index(_STATUS_LABEL))
    return eeg


def _is_biosemi(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(_BIOSEMI_MAGIC)) == _BIOSEMI_MAGIC


def _channel_label(loc: Any) -> str:
    if isinstance(loc, dict):
        return str(loc.get("labels", "")).strip()
    return str(loc).strip()


def _blockrange_seconds(blockrange: Any) -> tuple[float, float]:
    values = np.asarray(blockrange, dtype=float).reshape(-1)
    return float(values[0]), float(values[1])


def _status_events(status: np.ndarray, start_sample: int, stop_sample: int) -> list[dict[str, Any]]:
    words = np.asarray(status, dtype=np.int64) & 0xFFFFFF
    triggers = words & _TRIGGER_MASK
    triggers[(words & _EPOCH_BIT) != 0] = 0
    changes = np.diff(np.r_[0, triggers])
    positions = np.flatnonzero(changes > 0)
    selected = (positions >= start_sample) & (positions < stop_sample)
    return [
        {"type": int(triggers[position]), "latency": int(position - start_sample + 1)}
        for position in positions[selected]
    ]


def _replace_events(eeg: dict[str, Any], events: list[dict[str, Any]]) -> None:
    urevents = [dict(event) for event in events]
    linked = []
    for index, event in enumerate(events):
        record = dict(event)
        record["urevent"] = index
        linked.append(record)
    eeg["event"] = linked
    eeg["urevent"] = urevents


def _drop_channel(eeg: dict[str, Any], index: int) -> None:
    data = np.asarray(eeg["data"])
    keep = np.ones(data.shape[0], dtype=bool)
    keep[index] = False
    eeg["data"] = data[keep]
    eeg["nbchan"] = int(np.count_nonzero(keep))
    locations = [loc for channel, loc in enumerate(list(eeg["chanlocs"])) if channel != index]
    if isinstance(eeg["chanlocs"], np.ndarray):
        eeg["chanlocs"] = np.asarray(locations, dtype=object)
    else:
        eeg["chanlocs"] = locations

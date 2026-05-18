"""Shared File-menu import/export helpers."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import mne
import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


TEXT_EXTENSIONS = {".asc", ".csv", ".dat", ".text", ".tsv", ".txt"}
MATLAB_EXTENSIONS = {".mat"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
FLOAT32_FORMATS = {"float32", "float32le", "float32be"}


def infer_dataformat(filename: str | Path | None, dataformat: str | None = None) -> str:
    """Infer an EEGLAB-style import data format from a filename."""
    if dataformat and dataformat != "auto":
        return dataformat.lower()
    suffix = Path(filename or "").suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return "ascii"
    if suffix in MATLAB_EXTENSIONS:
        return "matlab"
    if suffix in NUMPY_EXTENSIONS:
        return suffix[1:]
    if suffix == ".fdt":
        return "float32le"
    return "ascii"


def load_data_array(
    data: Any,
    *,
    dataformat: str = "auto",
    nbchan: int | None = None,
    variable: str | None = None,
) -> np.ndarray:
    """Load a data array from an EEGLAB-style ``pop_importdata`` source."""
    if isinstance(data, (str, Path)):
        filename = Path(data)
        resolved_format = infer_dataformat(filename, dataformat)
        if resolved_format == "ascii":
            delimiter = "," if filename.suffix.lower() == ".csv" else None
            return np.loadtxt(filename, delimiter=delimiter)
        if resolved_format in {"matlab", "mat"}:
            mat = scipy.io.loadmat(filename, squeeze_me=True, struct_as_record=False)
            if variable:
                return np.asarray(mat[variable])
            keys = [key for key in mat if not key.startswith("__")]
            if not keys:
                raise ValueError(f"No MATLAB variables found in {filename}")
            return np.asarray(mat["data"] if "data" in mat else mat[keys[0]])
        if resolved_format == "npy":
            return np.asarray(np.load(filename, allow_pickle=False))
        if resolved_format == "npz":
            with np.load(filename, allow_pickle=False) as archive:
                key = variable or ("data" if "data" in archive else archive.files[0])
                return np.asarray(archive[key])
        if resolved_format in FLOAT32_FORMATS:
            dtype = ">f4" if resolved_format == "float32be" else "<f4"
            values = np.fromfile(filename, dtype=dtype)
            if nbchan and nbchan > 0:
                if values.size % nbchan:
                    raise ValueError("float32 file length is not divisible by nbchan")
                return values.reshape((int(nbchan), values.size // int(nbchan)))
            return values[np.newaxis, :]
        raise ValueError(f"Unsupported import data format: {resolved_format}")
    return np.asarray(data)


def eeg_from_data(
    data: Any,
    *,
    srate: float = 1.0,
    setname: str = "",
    nbchan: int | None = None,
    pnts: int | None = None,
    xmin: float = 0.0,
    subject: str = "",
    condition: str = "",
    group: str = "",
    session: int | str | None = None,
    comments: str = "",
    ref: str | list[int] = "common",
    chanlocs: list[dict[str, Any]] | np.ndarray | None = None,
    filename: str = "",
    filepath: str = "",
) -> dict[str, Any]:
    """Build a checked EEGLAB-like EEG dict from channel-major data."""
    array = np.asarray(data)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim not in {2, 3}:
        raise ValueError("EEG data must be 1-D, 2-D, or 3-D")
    if nbchan and array.shape[0] != int(nbchan):
        if array.ndim == 2 and array.shape[1] == int(nbchan):
            array = array.T
        else:
            raise ValueError("nbchan does not match imported data")
    if not nbchan and array.ndim == 2 and array.shape[0] > array.shape[1]:
        array = array.T
    if pnts and array.ndim == 2:
        pnts = int(pnts)
        if pnts > 0 and array.shape[1] % pnts == 0:
            array = array.reshape((array.shape[0], pnts, array.shape[1] // pnts))
    nbchan = int(array.shape[0])
    pnts = int(array.shape[1])
    trials = int(array.shape[2]) if array.ndim == 3 else 1
    srate = float(srate)
    xmax = float(xmin) + ((pnts - 1) / srate if pnts and srate else 0.0)
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": setname,
            "filename": filename,
            "filepath": filepath,
            "subject": subject,
            "condition": condition,
            "group": group,
            "session": [] if session is None or session == "" else session,
            "comments": comments,
            "nbchan": nbchan,
            "pnts": pnts,
            "trials": trials,
            "srate": srate,
            "xmin": float(xmin),
            "xmax": xmax,
            "times": np.linspace(float(xmin) * 1000, xmax * 1000, pnts) if pnts else np.array([]),
            "data": array,
            "chanlocs": _default_chanlocs(nbchan) if chanlocs is None else chanlocs,
            "ref": ref,
            "saved": "no",
        }
    )
    with strict_mode(False):
        return eeg_checkset(eeg)


def read_table_records(
    filename: str | Path,
    *,
    fields: list[str] | tuple[str, ...] | None = None,
    skipline: int = 0,
) -> list[dict[str, Any]]:
    """Read a simple CSV/TSV/whitespace table into record dictionaries."""
    path = Path(filename)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in _read_rows(stream, path.suffix.lower()) if row]
    rows = rows[int(skipline) :]
    if not rows:
        return []
    field_names = list(fields or [])
    if not field_names:
        first = rows[0]
        if any(not _is_float(value) for value in first):
            field_names = [str(value).strip() for value in first]
            rows = rows[1:]
        else:
            field_names = ["type", "latency", *[f"field{index}" for index in range(3, len(first) + 1)]]
    if any(len(row) != len(field_names) for row in rows):
        raise ValueError("All rows must have the same number of values as field names")
    return [dict(zip(field_names, (_coerce_value(value) for value in row))) for row in rows]


def records_to_events(
    records: list[dict[str, Any]],
    *,
    srate: float,
    timeunit: float | None = None,
) -> list[dict[str, Any]]:
    """Convert table records with a latency field to EEGLAB event dicts."""
    events = []
    for record in records:
        if "latency" not in record:
            raise ValueError("Imported event records must include a latency field")
        event = dict(record)
        event["latency"] = _latency_to_samples(event["latency"], srate=srate, timeunit=timeunit)
        if "type" not in event:
            event["type"] = "event"
        events.append(event)
    events.sort(key=lambda item: float(item.get("latency", math.inf)))
    return events


def events_to_records(events: Any) -> list[dict[str, Any]]:
    """Normalize EEG event containers to a list of dictionaries."""
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, dict):
        return [dict(events)]
    return [dict(event) for event in events]


def mne_raw_to_eeg(raw: mne.io.BaseRaw, *, setname: str = "", filename: str = "") -> dict[str, Any]:
    """Convert an MNE Raw object to an EEGPrep EEG dict."""
    try:
        data = raw.get_data(units="uV")
    except TypeError:
        data = raw.get_data() * 1_000_000.0
    eeg = eeg_from_data(
        data,
        srate=float(raw.info["sfreq"]),
        setname=setname,
        chanlocs=[{"labels": name, "type": "EEG"} for name in raw.ch_names],
        filename=Path(filename).name if filename else "",
        filepath=str(Path(filename).parent) if filename else "",
    )
    annotations = raw.annotations
    if annotations is not None and len(annotations):
        eeg["event"] = [
            {
                "type": str(annotation["description"]),
                "latency": float(annotation["onset"]) * float(raw.info["sfreq"]) + 1,
                "duration": float(annotation["duration"]) * float(raw.info["sfreq"]),
            }
            for annotation in annotations
        ]
    return eeg


def eeg_to_mne_raw(eeg: dict[str, Any]) -> mne.io.RawArray:
    """Convert a continuous EEGPrep EEG dict to MNE RawArray."""
    data = np.asarray(eeg["data"])
    if data.ndim == 3:
        data = data.reshape((data.shape[0], data.shape[1] * data.shape[2]))
    ch_names = channel_labels(eeg)
    info = mne.create_info(ch_names=ch_names, sfreq=float(eeg["srate"]), ch_types="eeg")
    raw = mne.io.RawArray(data / 1_000_000.0, info, verbose=False)
    events = events_to_records(eeg.get("event"))
    if events:
        onset = []
        duration = []
        description = []
        for event in events:
            latency = float(event.get("latency", 1))
            onset.append((latency - 1) / float(eeg["srate"]))
            duration.append(float(event.get("duration", 0) or 0) / float(eeg["srate"]))
            description.append(str(event.get("type", "event")))
        raw.set_annotations(mne.Annotations(onset, duration, description))
    return raw


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON using EEGPrep-safe conversion for NumPy containers."""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(json_safe(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def json_safe(value: Any) -> Any:
    """Convert common NumPy/Path containers to JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def _read_rows(stream: Any, suffix: str) -> list[list[str]]:
    if suffix == ".csv":
        return [row for row in csv.reader(stream)]
    if suffix == ".tsv":
        return [row for row in csv.reader(stream, delimiter="\t")]
    return [line.split() for line in stream if line.strip()]


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _coerce_value(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    try:
        numeric = float(value)
    except ValueError:
        return value
    return int(numeric) if numeric.is_integer() else numeric


def _latency_to_samples(value: Any, *, srate: float, timeunit: float | None) -> float:
    latency = float(value)
    if timeunit is None or (isinstance(timeunit, float) and math.isnan(timeunit)):
        return latency
    return latency * float(timeunit) * float(srate) + 1


def _default_chanlocs(nbchan: int) -> list[dict[str, Any]]:
    return [{"labels": f"Ch{index}", "type": "EEG"} for index in range(1, int(nbchan) + 1)]


def channel_labels(eeg: dict[str, Any]) -> list[str]:
    """Return channel labels for an EEGLAB-like EEG dict."""
    chanlocs = eeg.get("chanlocs")
    if chanlocs is None:
        chanlocs = []
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    labels = []
    for index in range(int(eeg.get("nbchan", 0))):
        if index < len(chanlocs) and isinstance(chanlocs[index], dict):
            labels.append(str(chanlocs[index].get("labels") or f"Ch{index + 1}"))
        else:
            labels.append(f"Ch{index + 1}")
    return labels

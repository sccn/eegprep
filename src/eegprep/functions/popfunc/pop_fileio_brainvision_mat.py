"""Import BrainVision Analyzer MATLAB exports.

The file layout follows ``pop_loadbva.m`` in the EEGLAB bva-io plugin
(https://github.com/sccn/bva-io/blob/master/pop_loadbva.m, lines 46-139):
one variable per channel, plus ``Channels``, ``SampleRate``, ``t``,
``SegmentCount``, and ``Markers``. History stays
``pop_fileio_brainvision_mat`` because that is the EEGPrep menu command.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_loadset import _is_hdf5_file
from eegprep.functions.sigprocfunc.convertlocs import convertlocs


logger = logging.getLogger(__name__)

_SKIPPED_FIELDS = frozenset({"t", "markercount", "markers", "samplerate", "segmentcount", "channelcount", "channels"})
_REQUIRED_FIELDS = ("Channels", "SampleRate", "SegmentCount", "t")


def pop_fileio_brainvision_mat(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import a BrainVision Analyzer MATLAB export.

    Parameters
    ----------
    filename : str or Path
        MATLAB v7 ``.mat`` export. v7.3 (HDF5) files are refused.
    return_com : bool, optional
        When true, also return the history command.

    Returns
    -------
    dict or tuple
        EEG dataset, or ``(EEG, command)`` when ``return_com`` is true.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise ValueError(
            f"pop_fileio_brainvision_mat does not accept {unknown}. Call it with the Analyzer .mat path only."
        )
    path = Path(filename)
    if path.suffix.lower() != ".mat":
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because the extension is not .mat. "
            "Export a BrainVision Analyzer MATLAB file, or import a .vhdr file with pop_loadbv."
        )
    eeg = _load_brainvision_analyzer_mat(path)
    command = f"EEG = pop_fileio_brainvision_mat({format_history_value(path)});"
    history = str(eeg.get("history") or "").strip()
    eeg["history"] = f"{history}\n{command}".lstrip()
    return (eeg, command) if return_com else eeg


def _load_brainvision_analyzer_mat(path: Path) -> dict[str, Any]:
    raw = _read_mat(path)
    missing = [name for name in _REQUIRED_FIELDS if not _has_field(raw, name)]
    if missing:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because {', '.join(missing)} is missing. "
            "Re-export from BrainVision Analyzer as a MATLAB file with one variable per channel "
            "plus Channels, SampleRate, t, SegmentCount, and Markers, or import the .vhdr with pop_loadbv."
        )
    channel_records = [_record_mapping(record) for record in _struct_records(_field(raw, "Channels"))]
    if not channel_records:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because Channels has no records. "
            "Re-export from BrainVision Analyzer, or import the .vhdr with pop_loadbv."
        )
    names = [_text(_record_value(record, "Name")).lower() for record in channel_records]
    segmented = _number(_field(raw, "SegmentCount")) > 1
    samples: dict[int, np.ndarray] = {}
    for field_name, value in raw.items():
        if field_name.lower() in _SKIPPED_FIELDS:
            continue
        index = _channel_index(field_name, names)
        if index is None:
            # pop_loadbva.m:61-65
            logger.warning("channel '%s' is not in the Channels structure", field_name.lower())
            names.append(field_name.lower())
            channel_records.append({"Name": field_name, "Phi": None, "Theta": None, "Radius": 0})
            index = len(names) - 1
        column = _channel_samples(value, segmented=segmented, field_name=field_name, path=path)
        previous = samples.get(index)
        if previous is not None and previous.shape != column.shape:
            raise ValueError(
                f"pop_fileio_brainvision_mat cannot import {path} because channel variables "
                f"do not share a shape ({previous.shape} vs {column.shape}). "
                "Re-export the Analyzer MATLAB file, or import the .vhdr with pop_loadbv."
            )
        samples[index] = column
    if not samples:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because it has no channel variables. "
            "Re-export from BrainVision Analyzer with one variable per channel, or import the .vhdr with pop_loadbv."
        )
    shapes = {column.shape for column in samples.values()}
    if len(shapes) != 1:
        shown = ", ".join(str(shape) for shape in sorted(shapes))
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because channel variables "
            f"do not share a shape ({shown}). "
            "Re-export the Analyzer MATLAB file, or import the .vhdr with pop_loadbv."
        )
    template = next(iter(samples.values()))
    data = np.zeros((len(channel_records), *template.shape), dtype=np.float64)
    for index, column in samples.items():
        data[index] = column
    pnts = int(data.shape[1])
    trials = int(data.shape[2]) if data.ndim == 3 else 1
    time_ms = np.asarray(_field(raw, "t"), dtype=np.float64).ravel()
    if time_ms.size == 0:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because t is empty. "
            "Re-export the Analyzer MATLAB file with the time axis in milliseconds."
        )
    srate = _number(_field(raw, "SampleRate"))
    xmin = float(time_ms[0]) / 1000.0
    xmax = float(time_ms[-1]) / 1000.0
    events = _events(_field(raw, "Markers"), segmented=segmented, pnts=pnts)
    for index, event in enumerate(events):
        event["urevent"] = index
    urevents = [{key: value for key, value in event.items() if key != "urevent"} for event in events]
    chanlocs = _chanlocs(channel_records)
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": "Brain Vision Analyzer file",
            "filename": "",
            "filepath": "",
            "comments": f"Original file: {path}",
            "nbchan": int(data.shape[0]),
            "pnts": pnts,
            "trials": trials,
            "srate": srate,
            "xmin": xmin,
            "xmax": xmax,
            "times": linspace(xmin * 1000.0, xmax * 1000.0, pnts),
            "data": data,
            "chanlocs": chanlocs,
            "event": events,
            "urevent": urevents,
            "saved": "no",
        }
    )
    return eeg_checkset(eeg, "eventconsistency")


def _read_mat(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because the file does not exist. "
            "Choose an existing Analyzer .mat export."
        )
    if _is_hdf5_file(path):
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because it is a MATLAB v7.3 (HDF5) file. "
            "Re-export it from BrainVision Analyzer as a MATLAB v7 .mat file, or import the .vhdr with pop_loadbv."
        )
    try:
        loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False, appendmat=False)
    except Exception as exc:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because it is not a MATLAB v7 file ({exc}). "
            "Re-export it from BrainVision Analyzer, or import the .vhdr with pop_loadbv."
        ) from exc
    return {key: value for key, value in loaded.items() if not str(key).startswith("__")}


def _channel_index(field_name: str, names: list[str]) -> int | None:
    lowered = field_name.lower()
    exact = [index for index, name in enumerate(names) if name == lowered]
    # pop_loadbva.m:55 matches the field, then the field without its first character.
    prefix = [index for index, name in enumerate(names) if len(lowered) > 1 and name == lowered[1:]]
    matches = exact + prefix
    if not matches:
        return None
    return matches[0]


def _channel_samples(value: Any, *, segmented: bool, field_name: str, path: Path) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    squeezed = np.squeeze(matrix)
    if segmented:
        if squeezed.ndim == 1:
            squeezed = squeezed.reshape(1, -1)
        if squeezed.ndim != 2:
            raise ValueError(
                f"pop_fileio_brainvision_mat cannot import {path} because channel {field_name} "
                f"has shape {matrix.shape} and SegmentCount is greater than 1. "
                "Each segmented channel must be a segments-by-time matrix."
            )
        # pop_loadbva.m:68 assigns the transpose into EEG.data(channel, :, :),
        # so rows of the transpose are time points and columns are segments.
        return np.array(squeezed.T, dtype=np.float64, copy=True)
    if squeezed.ndim > 1:
        raise ValueError(
            f"pop_fileio_brainvision_mat cannot import {path} because channel {field_name} "
            f"has shape {matrix.shape} and SegmentCount is 1. "
            "Each continuous channel must be a vector, or set SegmentCount to the number of segments."
        )
    return np.array(squeezed, dtype=np.float64, copy=True).ravel()


def _events(markers: Any, *, segmented: bool, pnts: int) -> list[dict[str, Any]]:
    if markers is None:
        return []
    events: list[dict[str, Any]] = []
    # pop_loadbva.m:110-113
    for epoch_index, record in _marker_rows(markers):
        mapping = _record_mapping(record)
        if _is_blank(_record_value(mapping, "Description")):
            continue
        position = _number(_record_value(mapping, "Position"))
        event: dict[str, Any] = {
            "type": _text(_record_value(mapping, "Description")),
            "latency": position,
            "description": _text(_record_value(mapping, "Description")),
        }
        for source, target in (
            ("Points", "Points"),
            ("Type", "bvatype"),
            ("Chan", "chan"),
            ("ChannelNumber", "channelnumber"),
        ):
            if _has_field(mapping, source):
                event[target] = _stored_value(_record_value(mapping, source))
        if segmented:
            event["epoch"] = epoch_index
            event["latency"] = position + (epoch_index - 1) * pnts
        events.append(event)
    return events


def _chanlocs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chanlocs: list[dict[str, Any]] = []
    for record in records:
        loc: dict[str, Any] = {"labels": _text(_record_value(record, "Name"))}
        radius = _optional_number(_record_value(record, "Radius"))
        if radius not in (None, 0.0):
            loc["sph_theta_besa"] = _optional_number(_record_value(record, "Theta"))
            loc["sph_phi_besa"] = _optional_number(_record_value(record, "Phi"))
            loc["sph_radius"] = radius
        chanlocs.append(loc)
    # pop_loadbva.m:101-103
    converted = convertlocs(chanlocs, "sphbesa2all")
    for loc in converted:
        loc.pop("sph_theta_besa", None)
        loc.pop("sph_phi_besa", None)
        loc.setdefault("ref", "")
        loc.setdefault("type", "")
    return converted


def _marker_rows(markers: Any) -> list[tuple[int, Any]]:
    array = np.asarray(markers, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array.reshape(-1, 1)
    rows = int(array.shape[0])
    columns = int(array.shape[1]) if array.ndim > 1 else 1
    flat = array.ravel(order="F")
    paired: list[tuple[int, Any]] = []
    for index1 in range(1, rows + 1):
        for index2 in range(columns):
            paired.append((index1, flat[index2 * rows + index1 - 1]))
    return paired


def _struct_records(value: Any) -> list[Any]:
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return [array.item()]
    return list(array.ravel(order="F"))


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    names = list(getattr(record, "_fieldnames", []))
    return {name: getattr(record, name) for name in names}


def _record_value(record: dict[str, Any], name: str) -> Any:
    for key, value in record.items():
        if str(key).lower() == name.lower():
            return value
    return None


def _has_field(record: dict[str, Any], name: str) -> bool:
    return any(str(key).lower() == name.lower() for key in record)


def _field(raw: dict[str, Any], name: str) -> Any:
    for key, value in raw.items():
        if str(key).lower() == name.lower():
            return value
    return None


def _stored_value(value: Any) -> Any:
    number = _optional_number(value)
    if number is not None:
        return number
    return _text(value)


def _optional_number(value: Any) -> float | None:
    if _is_blank(value):
        return None
    try:
        return _number(value)
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float:
    unwrapped = _unwrap(value)
    if isinstance(unwrapped, np.ndarray):
        if unwrapped.size != 1:
            raise ValueError(f"expected one number, got shape {unwrapped.shape}")
        unwrapped = unwrapped.reshape(-1)[0]
    return float(unwrapped)


def _text(value: Any) -> str:
    unwrapped = _unwrap(value)
    if unwrapped is None:
        return ""
    if isinstance(unwrapped, bytes):
        return unwrapped.decode("utf-8", errors="replace")
    if isinstance(unwrapped, np.ndarray):
        if unwrapped.size == 0:
            return ""
        if unwrapped.dtype.kind in "SU":
            return "".join(str(item) for item in unwrapped.reshape(-1).tolist())
        if unwrapped.size == 1:
            return _text(unwrapped.reshape(-1)[0])
    return str(unwrapped)


def _is_blank(value: Any) -> bool:
    unwrapped = _unwrap(value)
    if unwrapped is None:
        return True
    if isinstance(unwrapped, str):
        return unwrapped == ""
    if isinstance(unwrapped, bytes):
        return unwrapped == b""
    if isinstance(unwrapped, np.ndarray):
        return unwrapped.size == 0 or (unwrapped.dtype.kind in "SU" and _text(unwrapped) == "")
    return False


def _unwrap(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return _unwrap(value.reshape(-1)[0])
    return value


__all__ = ["pop_fileio_brainvision_mat"]

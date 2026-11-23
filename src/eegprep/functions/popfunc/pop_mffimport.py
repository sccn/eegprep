"""Import an EGI MFF recording."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.extras import import_optional_dependency
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._file_io import mne_raw_to_eeg
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset

_SYNTHETIC_CHANNELS = {"stm+", "resp", "fbk+", "lrnd", "stim"}
_TYPE_FIELDS = ("code", "label", "mffkeys")
# EGI's MFF XML schema stores epoch and event times in nanoseconds unless the
# element declares a different unit. mffmatlabio's Java layer divides version-0
# values by 1000 before the same /1e6 conversion, which is this /1e9.
_TIME_DIVISOR = {
    "ns": 1e9,
    "nanosecond": 1e9,
    "nanoseconds": 1e9,
    "us": 1e6,
    "µs": 1e6,
    "microsecond": 1e6,
    "microseconds": 1e6,
    "ms": 1e3,
    "millisecond": 1e3,
    "milliseconds": 1e3,
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "seconds": 1.0,
}
_MICROVOLTS_PER_VOLT = 1e6


def pop_mffimport(
    filename: str | Path,
    *,
    typefield: str = "code",
    gui: bool = False,
    renderer: Any = None,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import an EGI ``.mff`` folder.

    Segmented files stay epoched. Continuous files use the same MNE reader as
    File-IO. Synthetic stimulus channels (``stm+``, ``resp``, ``fbk+``,
    ``LRND``) are dropped. Cancel on the type-field dialog leaves the caller
    unchanged.
    """
    path = Path(filename)
    if not path.exists():
        raise ValueError(f"pop_mffimport: {path} does not exist. Choose an EGI .mff folder with File > Import data.")
    chosen = _type_choice(typefield)
    if gui or renderer is not None:
        result = inputgui(pop_mffimport_dialog_spec(), renderer=renderer)
        if result is None:
            return ({}, "") if return_com else {}
        index = int(result.get("typefield") or 1)
        if index < 1 or index > len(_TYPE_FIELDS):
            raise ValueError(
                f"pop_mffimport: type field choice {index} is outside 1..{len(_TYPE_FIELDS)}. "
                "Choose code, label, or mffkeys, then import again."
            )
        chosen = _TYPE_FIELDS[index - 1]
    mne = import_optional_dependency("mne", feature="pop_mffimport")
    raw = mne.io.read_raw_egi(path, preload=True, verbose="error")
    if _looks_segmented(path):
        eeg = _epochs_from_categories(raw, path, chosen)
    else:
        eeg = mne_raw_to_eeg(raw, setname=path.stem, filename=str(path))
    eeg = _drop_synthetic_channels(eeg)
    eeg["setname"] = eeg.get("setname") or path.stem
    eeg["comments"] = f"Original file: {path}"
    command = f"EEG = pop_mffimport({format_history_value(path)}, 'typefield', {format_history_value(chosen)});"
    history = str(eeg.get("history") or "").strip()
    eeg["history"] = f"{history}\n{command}".lstrip()
    return (eeg, command) if return_com else eeg


def pop_mffimport_dialog_spec() -> DialogSpec:
    """Return the event-type field list from ``pop_mffimport.m``."""
    return DialogSpec(
        title="Import EGI MFF -- pop_mffimport()",
        function_name="pop_mffimport",
        help_text="pop_mffimport",
        eeglab_source="plugins/mffmatlabio/pop_mffimport.m",
        geometry=((1,), (1,)),
        controls=(
            ControlSpec("text", "Event type field"),
            ControlSpec("popupmenu", "code|label|mffkeys", tag="typefield", value=1),
        ),
    )


def _looks_segmented(path: Path) -> bool:
    folder = path if path.is_dir() else path.parent
    return (folder / "categories.xml").is_file() or (folder / "Categories.xml").is_file()


def _type_choice(value: str) -> str:
    text = str(value or "code").strip().lower()
    if text not in _TYPE_FIELDS:
        shown = ", ".join(_TYPE_FIELDS)
        raise ValueError(
            f"pop_mffimport: typefield {value!r} is not one of {shown}. "
            "Pass typefield='code', 'label', or 'mffkeys', then import again."
        )
    return text


def _epochs_from_categories(raw: Any, path: Path, typefield: str) -> dict[str, Any]:
    segments = _mff_segments(path)
    if not segments:
        raise ValueError(
            f"pop_mffimport: {path} looks segmented but categories.xml has no segments. "
            "Export the epochs from Net Station again, then import the .mff folder."
        )
    continuous = np.asarray(raw.get_data(), dtype=np.float64) * _MICROVOLTS_PER_VOLT
    srate = float(raw.info["sfreq"])
    pieces = []
    cursor = 0
    for trial, segment in enumerate(segments, start=1):
        # MNE concatenates the epoch samples and drops the gaps, so the slice
        # follows epoch order rather than the absolute timeline.
        length = int(round((segment["end"] - segment["begin"]) * srate))
        last = cursor + length
        if length <= 0 or cursor < 0 or last > continuous.shape[1]:
            raise ValueError(
                f"pop_mffimport: segment {trial} of {path.name} spans samples {cursor}:{last}, "
                f"outside 0:{continuous.shape[1]}. Check epochs.xml and categories.xml, then import again."
            )
        pieces.append(continuous[:, cursor:last])
        cursor = last
    lengths = {piece.shape[1] for piece in pieces}
    if len(lengths) != 1:
        shown = ", ".join(str(length) for length in sorted(lengths))
        raise ValueError(
            f"pop_mffimport: {path.name} segments have different lengths ({shown}). "
            "EEGLAB epochs need one length. Re-segment the recording, then import again."
        )
    data = np.stack(pieces, axis=2)
    pnts = int(data.shape[1])
    first = segments[0]
    xmin = 0.0
    if first.get("event_begin") is not None:
        xmin = -(float(first["event_begin"]) - float(first["begin"]))
    events = _track_events(path, srate, segments, pnts, typefield)
    times = np.arange(pnts) / srate + xmin
    names = list(raw.ch_names)
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": path.stem,
            "filename": path.name,
            "filepath": str(path.parent if path.is_file() else path),
            "nbchan": int(data.shape[0]),
            "pnts": pnts,
            "trials": int(data.shape[2]),
            "srate": srate,
            "xmin": float(xmin),
            "xmax": float(times[-1]) if times.size else 0.0,
            "times": times * 1000.0,
            "data": data,
            "chanlocs": [{"labels": name, "type": "EEG"} for name in names],
            "event": events,
        }
    )
    return eeg


def _mff_segments(path: Path) -> list[dict[str, Any]]:
    folder = path if path.is_dir() else path.parent
    categories = _first_xml(folder, ("categories.xml", "Categories.xml"))
    epochs = _first_xml(folder, ("epochs.xml", "Epochs.xml"))
    bounds = _epoch_bounds(epochs) if epochs is not None else []
    segments: list[dict[str, Any]] = []
    if categories is not None:
        for category in categories.iter():
            if _xml_name(category.tag) != "cat":
                continue
            label = _child_text(category, "name")
            for segment in category.iter():
                if _xml_name(segment.tag) not in {"segment", "seg"}:
                    continue
                begin = _declared_seconds(segment, "beginTime")
                end = _declared_seconds(segment, "endTime")
                if bounds and len(segments) < len(bounds):
                    begin, end = bounds[len(segments)]
                if begin is None or end is None:
                    continue
                segments.append(
                    {
                        "name": label,
                        "begin": begin,
                        "end": end,
                        "event_begin": _first_seconds(segment, ("eventBegin", "evtBegin")),
                        "keys": _xml_keys(segment),
                    }
                )
    if not segments and bounds:
        segments = [
            {"name": str(index), "begin": begin, "end": end, "event_begin": None, "keys": []}
            for index, (begin, end) in enumerate(bounds, start=1)
        ]
    return segments


def _epoch_bounds(root: ET.Element) -> list[tuple[float, float]]:
    bounds = []
    for epoch in root.iter():
        if _xml_name(epoch.tag) != "epoch":
            continue
        begin = _declared_seconds(epoch, "beginTime")
        end = _declared_seconds(epoch, "endTime")
        if begin is None or end is None:
            continue
        bounds.append((begin, end))
    return bounds


def _first_xml(folder: Path, names: tuple[str, ...]) -> ET.Element | None:
    for name in names:
        file = folder / name
        if file.is_file():
            return ET.parse(file).getroot()
    return None


def _xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(node: ET.Element, name: str) -> str:
    for child in list(node):
        if _xml_name(child.tag) == name:
            return str(child.text or "").strip()
    return ""


def _declared_seconds(node: ET.Element, name: str) -> float | None:
    """Convert one MFF time element using the unit it declares.

    The EGI schema default is nanoseconds. A ``unit`` or ``units`` attribute
    selects microseconds, milliseconds, or seconds instead.
    """
    for child in list(node):
        if _xml_name(child.tag) != name:
            continue
        text = str(child.text or "").strip()
        if not text:
            return None
        unit = str(child.attrib.get("unit") or child.attrib.get("units") or "ns").strip().lower()
        divisor = _TIME_DIVISOR.get(unit)
        if divisor is None:
            shown = ", ".join(sorted(set(_TIME_DIVISOR)))
            raise ValueError(
                f"pop_mffimport: <{name}> declares unit {unit!r}, which is not one of {shown}. "
                "Fix that unit in the MFF XML, then import again."
            )
        return float(text) / divisor
    return None


def _first_seconds(node: ET.Element, names: tuple[str, ...]) -> float | None:
    for name in names:
        value = _declared_seconds(node, name)
        if value is not None:
            return value
    return None


def _xml_keys(node: ET.Element) -> list[dict[str, str]]:
    keys = []
    for key in list(node):
        if _xml_name(key.tag) != "keys":
            continue
        for item in list(key):
            if _xml_name(item.tag) != "key":
                continue
            keys.append(_key_record(item))
    if keys:
        return keys
    for item in node.iter():
        if _xml_name(item.tag) == "key" and item is not node:
            keys.append(_key_record(item))
    return keys


def _key_record(item: ET.Element) -> dict[str, str]:
    return {
        "code": _child_text(item, "keyCode") or _child_text(item, "code"),
        "data": _child_text(item, "data"),
        "datatype": _child_text(item, "dataType") or _child_text(item, "datatype"),
        "description": _child_text(item, "description"),
    }


def _track_events(
    path: Path,
    srate: float,
    segments: list[dict[str, Any]],
    pnts: int,
    typefield: str,
) -> list[dict[str, Any]]:
    folder = path if path.is_dir() else path.parent
    record_time = _record_time(folder)
    sample_beg = [float(segment["begin"]) * srate for segment in segments]
    sample_len = [float(segment["end"] - segment["begin"]) * srate for segment in segments]
    events: list[dict[str, Any]] = []
    for track in sorted(folder.glob("*.xml")):
        if not track.name.lower().startswith("events"):
            continue
        root = ET.parse(track).getroot()
        track_name = _child_text(root, "name") or track.stem
        track_type = _child_text(root, "trackType")
        for node in root.iter():
            if _xml_name(node.tag) != "event":
                continue
            events.append(
                _track_event(
                    node,
                    srate=srate,
                    record_time=record_time,
                    sample_beg=sample_beg,
                    sample_len=sample_len,
                    pnts=pnts,
                    trials=len(segments),
                    typefield=typefield,
                    track_name=track_name,
                    track_type=track_type,
                    source=track.name,
                )
            )
    events.sort(key=lambda event: (float(event["latency"]), int(event["epoch"]), str(event["type"])))
    return events


def _track_event(
    node: ET.Element,
    *,
    srate: float,
    record_time: datetime | None,
    sample_beg: list[float],
    sample_len: list[float],
    pnts: int,
    trials: int,
    typefield: str,
    track_name: str,
    track_type: str,
    source: str,
) -> dict[str, Any]:
    code = _child_text(node, "code")
    label = _child_text(node, "label")
    keys = _xml_keys(node)
    relative = _first_seconds(node, ("relativeBeginTime", "relativebegintime"))
    begin_text = _child_text(node, "beginTime")
    if relative is not None and relative > 0:
        # mff_importevents.m uses relativebegintime/1e6*srate. XML values are
        # nanoseconds, so /1e9*srate is the same instant.
        latency = relative * srate
    elif begin_text and "T" not in begin_text:
        numeric = _first_seconds(node, ("beginTime",))
        latency = 0.0 if numeric is None else numeric * srate
    else:
        if record_time is None:
            raise ValueError(
                f"pop_mffimport: {source} event {code or label or 'without a code'} has no relativeBeginTime, "
                "and info.xml has no recordTime. Add recordTime, then import again."
            )
        latency = (_parse_mff_time(begin_text) - record_time).total_seconds() * srate
    latency = _correct_epoch_gap(latency, sample_beg, sample_len)
    duration_seconds = _first_seconds(node, ("duration",))
    duration = 0.0 if duration_seconds is None else duration_seconds * srate
    epoch = int(np.floor((latency + 1e-6) / pnts)) + 1 if pnts else 1
    epoch = min(max(epoch, 1), max(trials, 1))
    event: dict[str, Any] = {
        "code": code,
        "label": label,
        "description": _child_text(node, "description"),
        "duration": float(duration),
        "latency": float(latency),
        "epoch": epoch,
        "sourcedevice": _child_text(node, "sourceDevice"),
        "name": track_name,
        "tracktype": track_type,
        "type": _event_type(code, label, keys, typefield),
    }
    _attach_mff_keys(event, keys)
    return event


def _event_type(code: str, label: str, keys: list[dict[str, str]], typefield: str) -> str:
    if typefield == "label":
        return label or code or "event"
    if typefield == "mffkeys":
        return _mffkeys_text(keys) or code or label or "event"
    return code or label or "event"


def _attach_mff_keys(event: dict[str, Any], keys: list[dict[str, str]]) -> None:
    if not keys:
        return
    event["mffkeys"] = _mffkeys_text(keys)
    for key in keys:
        cleaned = "".join(character for character in key["code"] if ("0" <= character <= "9") or ord(character) >= 64)
        if cleaned:
            event[f"mffkey_{cleaned}"] = key["data"]


def _mffkeys_text(keys: list[dict[str, str]]) -> str:
    return ";".join(f"{key['code']}={key['data']}" for key in keys if key["code"] or key["data"])


def _correct_epoch_gap(latency: float, sample_beg: list[float], sample_len: list[float]) -> float:
    """Match ``mff_import.m``: subtract the gap before the epoch that holds the event."""
    earlier = [index for index, begin in enumerate(sample_beg) if latency > begin]
    if not earlier:
        return latency
    last = earlier[-1]
    correction = sample_beg[last] - sum(sample_len[:last])
    return latency - correction


def _record_time(folder: Path) -> datetime | None:
    info = _first_xml(folder, ("info.xml", "Info.xml"))
    if info is None:
        return None
    text = ""
    for node in info.iter():
        if _xml_name(node.tag) == "recordTime":
            text = str(node.text or "").strip()
            break
    if not text:
        return None
    return _parse_mff_time(text)


def _parse_mff_time(text: str) -> datetime:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    # A one-digit timezone offset (-8) is not ISO. Pad it the way mff_importevents.m does.
    if len(value) >= 2 and value[-2] == "-" and value[-1].isdigit():
        value = f"{value[:-1]}0{value[-1]}:00"
    elif len(value) >= 3 and value[-3] == "-" and value[-2:].isdigit():
        value = f"{value[:-2]}{value[-1]}:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"pop_mffimport: MFF time {text!r} is not an ISO date. "
            "Fix beginTime or info.xml recordTime, then import again."
        ) from exc


def _drop_synthetic_channels(eeg: dict[str, Any]) -> dict[str, Any]:
    labels = [str(channel.get("labels") or "") for channel in eeg.get("chanlocs") or []]
    keep = [index for index, label in enumerate(labels) if label.lower() not in _SYNTHETIC_CHANNELS]
    if len(keep) == len(labels):
        return eeg
    data = np.asarray(eeg["data"])
    eeg["data"] = data[keep]
    eeg["chanlocs"] = [eeg["chanlocs"][index] for index in keep]
    eeg["nbchan"] = len(keep)
    return eeg

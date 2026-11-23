"""Import an EGI MFF recording."""

from __future__ import annotations

import xml.etree.ElementTree as ET
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
    continuous = np.asarray(raw.get_data())
    srate = float(raw.info["sfreq"])
    pieces = []
    events: list[dict[str, Any]] = []
    for trial, segment in enumerate(segments, start=1):
        first = int(round(segment["begin"] * srate))
        last = int(round(segment["end"] * srate))
        if last <= first or first < 0 or last > continuous.shape[1]:
            raise ValueError(
                f"pop_mffimport: segment {trial} of {path.name} spans samples {first}:{last}, "
                f"outside 0:{continuous.shape[1]}. Check epochs.xml and categories.xml, then import again."
            )
        pieces.append(continuous[:, first:last])
    lengths = {piece.shape[1] for piece in pieces}
    if len(lengths) != 1:
        shown = ", ".join(str(length) for length in sorted(lengths))
        raise ValueError(
            f"pop_mffimport: {path.name} segments have different lengths ({shown}). "
            "EEGLAB epochs need one length. Re-segment the recording, then import again."
        )
    data = np.stack(pieces, axis=2)
    pnts = int(data.shape[1])
    for trial, segment in enumerate(segments, start=1):
        latency = float((trial - 1) * pnts + 1)
        events.append({"type": _segment_type(segment, typefield), "latency": latency, "epoch": trial})
        for key in segment["keys"]:
            events.append(
                {"type": _key_type(key, typefield), "latency": latency, "epoch": trial, "mffkeys": key["code"]}
            )
    times = np.arange(pnts) / srate
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
            "xmin": 0.0,
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
                keys = [
                    {"code": _child_text(key, "code"), "data": _child_text(key, "data")}
                    for key in segment.iter()
                    if _xml_name(key.tag) == "key"
                ]
                begin = _seconds(_child_text(segment, "beginTime"))
                end = _seconds(_child_text(segment, "endTime"))
                if bounds and len(segments) < len(bounds):
                    begin, end = bounds[len(segments)]
                segments.append({"name": label, "begin": begin, "end": end, "keys": keys})
    if not segments and bounds:
        segments = [
            {"name": str(index), "begin": begin, "end": end, "keys": []}
            for index, (begin, end) in enumerate(bounds, start=1)
        ]
    return segments


def _epoch_bounds(root: ET.Element) -> list[tuple[float, float]]:
    bounds = []
    for epoch in root.iter():
        if _xml_name(epoch.tag) != "epoch":
            continue
        bounds.append((_seconds(_child_text(epoch, "beginTime")), _seconds(_child_text(epoch, "endTime"))))
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


def _seconds(text: str) -> float:
    if not text:
        return 0.0
    value = float(text)
    if abs(value) >= 1000:
        return value / 1_000_000.0
    return value


def _segment_type(segment: dict[str, Any], typefield: str) -> str:
    if typefield == "code" and segment["keys"] and segment["keys"][0]["code"]:
        return str(segment["keys"][0]["code"])
    if typefield == "mffkeys" and segment["keys"]:
        return str(segment["keys"][0]["data"] or segment["keys"][0]["code"] or segment["name"])
    return str(segment["name"] or "TLE")


def _key_type(key: dict[str, str], typefield: str) -> str:
    if typefield == "label":
        return key["data"] or key["code"] or "key"
    return key["code"] or key["data"] or "key"


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

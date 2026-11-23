"""Import an EGI MFF recording."""

from __future__ import annotations

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
    if _looks_segmented(path):
        epochs = mne.read_epochs_egi(path, verbose="error")
        eeg = _epochs_to_eeg(epochs, path, chosen)
    else:
        raw = mne.io.read_raw_egi(path, preload=True, verbose="error")
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
    # pop_mffimport.m epochs only when categories exist. Continuous MFF folders
    # also contain epochs.xml, one entry per recording block.
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


def _epochs_to_eeg(epochs: Any, path: Path, typefield: str) -> dict[str, Any]:
    array = np.asarray(epochs.get_data())
    data = np.transpose(array, (1, 2, 0))
    times = np.asarray(epochs.times)
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": path.stem,
            "filename": path.name,
            "filepath": str(path.parent if path.is_file() else path),
            "nbchan": int(data.shape[0]),
            "pnts": int(data.shape[1]),
            "trials": int(data.shape[2]),
            "srate": float(epochs.info["sfreq"]),
            "xmin": float(times[0]) if times.size else 0.0,
            "xmax": float(times[-1]) if times.size else 0.0,
            "times": times * 1000.0,
            "data": data,
            "chanlocs": [{"labels": name, "type": "EEG"} for name in epochs.ch_names],
            "event": _epoch_events(epochs, typefield),
        }
    )
    return eeg


def _epoch_events(epochs: Any, typefield: str) -> list[dict[str, Any]]:
    events = []
    metadata = getattr(epochs, "metadata", None)
    # events[:, 0] is a sample in the original continuous file, not an offset
    # inside the epoch. The time-locking event sits at the zero-time sample.
    zero = int(np.argmin(np.abs(np.asarray(epochs.times)))) if np.asarray(epochs.times).size else 0
    for index, row in enumerate(np.asarray(epochs.events)):
        meta = metadata.iloc[index] if metadata is not None and index < len(metadata) else None
        event = {
            "type": _epoch_type(row, meta, typefield),
            "latency": float(index * epochs.times.size + zero + 1),
            "epoch": index + 1,
        }
        if metadata is not None and index < len(metadata):
            for key, value in metadata.iloc[index].items():
                event[str(key)] = value
        events.append(event)
    return events


def _epoch_type(row: Any, meta: Any, typefield: str) -> str:
    if meta is not None and typefield == "label":
        for key in ("label", "Label", "description"):
            if key in meta.index and str(meta[key]).strip():
                return str(meta[key])
    if meta is not None and typefield == "mffkeys":
        for key in ("mffkeys", "code", "label"):
            if key in meta.index and str(meta[key]).strip():
                return str(meta[key])
    return str(row[2])


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

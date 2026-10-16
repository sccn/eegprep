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
_SUPPORTED_BDF_EVENT_MODES = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 99})
_LOWER_16 = 0xFFFF
_LOWER_8 = 0xFF
_EPOCH_BIT = 0x10000
_FALL_BIT = 0x8000
_EPOCH_START = 0x7FFE
_EPOCH_END = 0xFFFE


def pop_biosig(
    filename: str | Path, *, return_com: bool = False, **kwargs: Any
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import BIOSIG-style EDF/BDF/GDF files, optionally over a time range.

    BioSemi BDF files decode the ``Status`` channel with ``bdfeventmode``
    (default 4: rising edges of the lower 16 bits). ``importevent`` (default
    ``on``) stores those triggers, and ``rmeventchan`` (default ``on``) removes
    ``Status`` from the data. Unsupported values raise ``ValueError``.
    """
    path = Path(filename)
    if path.suffix.lower() not in _BIOSIG_SUFFIXES:
        raise ValueError(
            "pop_biosig supports EDF, BDF, and GDF files. Use pop_fileio or pop_loadset for other formats."
        )
    history_options = dict(kwargs)
    event_mode = _bdf_event_mode(kwargs.pop("bdfeventmode", 4))
    remove_event_channel = _on_off("rmeventchan", kwargs.pop("rmeventchan", "on"))
    import_events = _on_off("importevent", kwargs.pop("importevent", "on"))
    blockrange = kwargs.get("blockrange")
    eeg, _command = pop_fileio(filename, return_com=True, **kwargs)
    if path.suffix.lower() == ".bdf":
        eeg = _import_bdf_status_channel(
            eeg,
            path,
            blockrange,
            event_mode=event_mode,
            import_events=import_events,
            remove_event_channel=remove_event_channel,
        )
    arguments = [format_history_value(path)]
    for key, value in history_options.items():
        arguments.extend([format_history_value(key), format_history_value(value)])
    command = f"EEG = pop_biosig({', '.join(arguments)});"
    eeg["history"] = command
    return (eeg, command) if return_com else eeg


def _bdf_event_mode(value: Any) -> int:
    """Return a supported BIOSIG ``bdf2biosig_events`` mode."""
    if isinstance(value, bool):
        raise ValueError("bdfeventmode must be an integer BDF status mode")
    if isinstance(value, (int, np.integer)):
        mode = int(value)
    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
        mode = int(value)
    else:
        raise ValueError("bdfeventmode must be an integer BDF status mode")
    if mode not in _SUPPORTED_BDF_EVENT_MODES:
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_BDF_EVENT_MODES))
        raise ValueError(f"bdfeventmode {mode} is not supported; supported modes are {supported}")
    return mode


def _on_off(name: str, value: Any) -> bool:
    """Return whether an EEGLAB ``on``/``off`` option is enabled."""
    if not isinstance(value, str) or value.lower() not in {"on", "off"}:
        raise ValueError(f"{name} must be 'on' or 'off'")
    return value.lower() == "on"


def _import_bdf_status_channel(
    eeg: dict[str, Any],
    path: Path,
    blockrange: Any,
    *,
    event_mode: int,
    import_events: bool,
    remove_event_channel: bool,
) -> dict[str, Any]:
    """Decode BioSemi Status triggers and optionally remove that channel.

    A ``blockrange`` reads only the records overlapping the requested samples.
    The preceding record is included when the window starts on a record
    boundary, so the edge at the first requested sample uses the real preceding
    status word. Mode 7 also reads the file's first status sample, because its
    active level is relative to that sample.
    """
    labels = [_channel_label(loc) for loc in list(eeg.get("chanlocs", []))]
    if _STATUS_LABEL not in labels or not _is_biosemi(path):
        return eeg
    if import_events:
        _replace_events(eeg, _blockrange_status_events(path, blockrange, event_mode))
    if remove_event_channel:
        _drop_channel(eeg, labels.index(_STATUS_LABEL))
    return eeg


def _blockrange_status_events(path: Path, blockrange: Any, event_mode: int) -> list[dict[str, Any]]:
    """Return Status events for the requested half-open sample window."""
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
        start_sample = max(0, int(start * sample_rate))
        stop_sample = min(total_samples, int(stop * sample_rate))
    if stop_sample <= start_sample:
        return []
    first_record, last_record = _status_record_span(start_sample, stop_sample, samples_per_record, record_count)
    decoded, _raw = readbdf(opened, np.arange(first_record, last_record + 1), 3)
    status = np.asarray(decoded["Record"][status_index], dtype=np.int64)
    origin = (first_record - 1) * samples_per_record
    anchor = _status_anchor(opened, status_index, status, origin) if event_mode == 7 else 0
    return _status_events(
        status,
        origin=origin,
        start_sample=start_sample,
        stop_sample=stop_sample,
        mode=event_mode,
        anchor=anchor,
    )


def _status_record_span(
    start_sample: int,
    stop_sample: int,
    samples_per_record: int,
    record_count: int,
) -> tuple[int, int]:
    """Return the inclusive 1-based records needed to decode a sample window."""
    first = start_sample // samples_per_record + 1
    if start_sample > 0 and start_sample % samples_per_record == 0:
        first -= 1
    last = (stop_sample - 1) // samples_per_record + 1
    return max(first, 1), min(last, record_count)


def _status_anchor(opened: dict[str, Any], status_index: int, status: np.ndarray, origin: int) -> int:
    """Return the file's first unsigned Status word for mode 7."""
    if origin == 0:
        return int(status.reshape(-1)[0]) & 0xFFFFFF if status.size else 0
    decoded, _raw = readbdf(opened, [1], 3)
    first = np.asarray(decoded["Record"][status_index], dtype=np.int64).reshape(-1)
    return int(first[0]) & 0xFFFFFF


def _status_events(
    status: np.ndarray,
    *,
    origin: int,
    start_sample: int,
    stop_sample: int,
    mode: int,
    anchor: int,
) -> list[dict[str, Any]]:
    """Return in-window events for one contiguous Status excerpt.

    ``origin`` is the 0-based file sample of ``status[0]``. Positions before
    ``start_sample`` are dropped, including an edge created where the excerpt
    itself begins after sample 0.
    """
    words = np.asarray(status, dtype=np.int64).reshape(-1) & 0xFFFFFF
    events = []
    for event_type, index in _decode_bdf_status(words, mode, anchor=anchor):
        position = origin + index
        if start_sample <= position < stop_sample:
            events.append({"type": event_type, "latency": int(position - start_sample + 1)})
    return events


def _decode_bdf_status(words: np.ndarray, mode: int, *, anchor: int) -> list[tuple[int, int]]:
    """Return ``(type, excerpt index)`` pairs for BIOSIG BDF modes 1-8 and 99.

    Modes 3 and 5 keep each trigger code on the edge that produced it. The
    edge index is the first sample of the new level, matching ``diff([0; t])``.
    """
    if words.size == 0:
        return []
    if mode == 1:
        groups = _mode1_groups(words)
    elif mode == 2:
        groups = _rising_groups(words & _LOWER_16)
    elif mode == 3:
        groups = _both_edge_groups(words, _LOWER_16, or_fall_bit=False)
    elif mode == 4:
        groups = _rising_groups(_trigger_values(words, _LOWER_16))
    elif mode == 5:
        groups = _both_edge_groups(words, _LOWER_8, or_fall_bit=False)
    elif mode == 6:
        groups = _rising_groups(_trigger_values(words, _LOWER_8))
    elif mode == 7:
        groups = _bit_groups(words, anchor=anchor, falling=True)
    elif mode == 8:
        groups = _bit_groups(words, anchor=0, falling=False)
    elif mode == 99:
        groups = _both_edge_groups_from_values(words & _LOWER_16, or_fall_bit=True)
    return _sorted_events(groups)


def _mode1_groups(words: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    epoch = words & _EPOCH_BIT
    lower = words & _LOWER_16
    epoch_changes = _changes(epoch)
    label_changes = _changes(lower)
    epoch_rises = np.flatnonzero(epoch_changes > 0)
    label_rises = np.flatnonzero(label_changes > 0)
    epoch_falls = np.flatnonzero(epoch_changes < 0)
    label_falls = np.flatnonzero(label_changes < 0)
    previous = np.r_[np.int64(0), lower[:-1]]
    return [
        (epoch_rises, np.full(epoch_rises.size, _EPOCH_START, dtype=np.int64)),
        (label_rises, lower[label_rises] & _LOWER_8),
        (epoch_falls, np.full(epoch_falls.size, _EPOCH_END, dtype=np.int64)),
        (label_falls, (previous[label_falls] & _LOWER_8) | _FALL_BIT),
    ]


def _trigger_values(words: np.ndarray, mask: int) -> np.ndarray:
    values = (words & mask).copy()
    values[(words & _EPOCH_BIT) != 0] = 0
    return values


def _both_edge_groups(words: np.ndarray, mask: int, *, or_fall_bit: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    return _both_edge_groups_from_values(_trigger_values(words, mask), or_fall_bit=or_fall_bit)


def _both_edge_groups_from_values(values: np.ndarray, *, or_fall_bit: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair each edge with its code.

    ``or_fall_bit`` combines the previous code with the falling-edge flag by
    bitwise or. Otherwise the flag is added, matching BIOSIG modes 3 and 5.
    """
    changes = _changes(values)
    rises = np.flatnonzero(changes > 0)
    falls = np.flatnonzero(changes < 0)
    previous = np.r_[np.int64(0), values[:-1]]
    fall_types = previous[falls] | _FALL_BIT if or_fall_bit else previous[falls] + _FALL_BIT
    return [(rises, values[rises]), (falls, fall_types)]


def _rising_groups(values: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    rises = np.flatnonzero(_changes(values) > 0)
    return [(rises, values[rises])]


def _bit_groups(words: np.ndarray, *, anchor: int, falling: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = []
    for bit_number in range(1, 17):
        mask = np.int64(1 << (bit_number - 1))
        bit = ((words & mask) != 0).astype(np.int64)
        if falling:
            anchor_bit = int((np.int64(anchor) & mask) != 0)
            bit = (bit != anchor_bit).astype(np.int64)
        changes = _changes(bit)
        rises = np.flatnonzero(changes > 0)
        groups.append((rises, np.full(rises.size, bit_number, dtype=np.int64)))
        if falling:
            falls = np.flatnonzero(changes < 0)
            groups.append((falls, np.full(falls.size, bit_number + _FALL_BIT, dtype=np.int64)))
    return groups


def _changes(values: np.ndarray) -> np.ndarray:
    return np.diff(np.r_[np.int64(0), np.asarray(values, dtype=np.int64)])


def _sorted_events(groups: list[tuple[np.ndarray, np.ndarray]]) -> list[tuple[int, int]]:
    positions = np.concatenate([np.asarray(pos, dtype=np.int64) for pos, _types in groups])
    types = np.concatenate([np.asarray(event_types, dtype=np.int64) for _pos, event_types in groups])
    if positions.size == 0:
        return []
    order = np.argsort(positions, kind="mergesort")
    return [(int(types[index]), int(positions[index])) for index in order]


def _is_biosemi(path: Path) -> bool:
    """Return whether ``path`` starts with the BioSemi BDF magic."""
    with path.open("rb") as stream:
        return stream.read(len(_BIOSEMI_MAGIC)) == _BIOSEMI_MAGIC


def _channel_label(loc: Any) -> str:
    if isinstance(loc, dict):
        return str(loc.get("labels", "")).strip()
    return str(loc).strip()


def _blockrange_seconds(blockrange: Any) -> tuple[float, float]:
    values = np.asarray(blockrange, dtype=float).reshape(-1)
    return float(values[0]), float(values[1])


def _replace_events(eeg: dict[str, Any], events: list[dict[str, Any]]) -> None:
    """Replace dataset events and give each new event a 0-based urevent index."""
    urevents = [dict(event) for event in events]
    linked = []
    for index, event in enumerate(events):
        record = dict(event)
        record["urevent"] = index
        linked.append(record)
    eeg["event"] = linked
    eeg["urevent"] = urevents


def _drop_channel(eeg: dict[str, Any], index: int) -> None:
    """Remove one channel from data and channel locations."""
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

"""Read legacy Neuroscan epoch-oriented ``.eeg`` files."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def loadeeg(
    filename: str | Path,
    chanlist: Any = "all",
    triallist: Any = "all",
    typerange: Any = "all",
    accepttype: Any = "all",
    rtrange: Any = "all",
    responsetype: Any = "all",
    format: str = "auto",
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    int,
    int,
    float,
    float,
    float,
]:
    """Load a Neuroscan epoch file and its per-sweep metadata.

    Channel and trial selections use EEGLAB-facing 1-based indices. Truncated
    files return every complete selected sweep and discard the incomplete tail.

    Returns:
        ``(signal, accept, types, reaction_times, responses, channel_names,
        points, sweeps, sampling_rate, xmin, xmax)``. Selected sweeps are
        concatenated along the signal's sample axis.
    """
    data_format = str(format).lower()
    if data_format not in {"short", "int32", "auto"}:
        raise ValueError("format must be 'short', 'int32', or 'auto'")
    path = Path(filename)
    with path.open("rb") as stream:
        if data_format == "auto":
            data_format = _detect_format(stream)
        header = _read_header(stream)
        names, baselines, factors = _read_electrodes(stream, header.channels)
        channels = _one_based_indices(chanlist, header.channels, "channel")
        selected_trials = _selection_set(triallist, header.sweeps, "trial")
        dtype = np.dtype("<i2" if data_format == "short" else "<i4")
        samples_per_sweep = header.channels * header.points

        signals: list[np.ndarray] = []
        accepts: list[int] = []
        types: list[int] = []
        reaction_times: list[float] = []
        responses: list[int] = []
        for sweep in range(1, header.sweeps + 1):
            metadata = _read_sweep_header(stream)
            if metadata is None:
                logger.warning("Neuroscan file ended before sweep %d header", sweep)
                break
            raw_bytes = stream.read(samples_per_sweep * dtype.itemsize)
            if len(raw_bytes) != samples_per_sweep * dtype.itemsize:
                logger.warning("Neuroscan file is truncated in sweep %d; incomplete data were discarded", sweep)
                break
            if not _selected_sweep(
                sweep,
                metadata,
                selected_trials=selected_trials,
                typerange=typerange,
                accepttype=accepttype,
                rtrange=rtrange,
                responsetype=responsetype,
            ):
                continue
            raw = np.frombuffer(raw_bytes, dtype=dtype).reshape((header.channels, header.points), order="F")
            scaled = (raw.astype(float) - baselines[:, None]) * factors[:, None]
            signals.append(scaled[channels])
            accepts.append(metadata.accept)
            types.append(metadata.type)
            reaction_times.append(metadata.reaction_time)
            responses.append(metadata.response)

    signal = np.concatenate(signals, axis=1) if signals else np.empty((len(channels), 0), dtype=float)
    return (
        signal,
        np.asarray(accepts, dtype=int),
        np.asarray(types, dtype=int),
        np.asarray(reaction_times, dtype=float),
        np.asarray(responses, dtype=int),
        [names[index] for index in channels],
        header.points,
        len(signals),
        header.sampling_rate,
        header.xmin,
        header.xmax,
    )


@dataclass(frozen=True)
class _Header:
    sweeps: int
    points: int
    channels: int
    sampling_rate: float
    xmin: float
    xmax: float


@dataclass(frozen=True)
class _SweepHeader:
    accept: int
    type: int
    reaction_time: float
    response: int


def _detect_format(stream: Any) -> str:
    original = stream.tell()
    try:
        stream.seek(12)
        raw = stream.read(4)
        if len(raw) != 4:
            return "short"
        next_file = struct.unpack("<i", raw)[0]
        if next_file <= 0:
            return "short"
        stream.seek(next_file + 52)
        return "int32" if stream.read(1) == b"\x01" else "short"
    finally:
        stream.seek(original)


def _read_header(stream: Any) -> _Header:
    try:
        stream.seek(0)
        _read_exact(stream, 20)
        _read_exact(stream, 342)
        sweeps = _unpack(stream, "H")
        _read_exact(stream, 4)
        points = _unpack(stream, "H")
        channels = _unpack(stream, "H")
        _read_exact(stream, 4)
        sampling_rate = float(_unpack(stream, "H"))
        _read_exact(stream, 127)
        xmin = float(_unpack(stream, "f"))
        xmax = float(_unpack(stream, "f"))
        _read_exact(stream, 387)
    except EOFError as exc:
        raise ValueError("Neuroscan file ended inside the general header") from exc
    if points <= 0 or channels <= 0 or sampling_rate <= 0:
        raise ValueError("Neuroscan header contains invalid dimensions or sampling rate")
    return _Header(sweeps, points, channels, sampling_rate, xmin, xmax)


def _read_electrodes(stream: Any, channel_count: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    baselines = np.empty(channel_count, dtype=float)
    factors = np.empty(channel_count, dtype=float)
    try:
        for channel in range(channel_count):
            label = _read_exact(stream, 10).split(b"\x00", 1)[0].decode("latin-1").strip()
            _read_exact(stream, 37)
            baselines[channel] = _unpack(stream, "H")
            _read_exact(stream, 10)
            sensitivity = float(_unpack(stream, "f"))
            _read_exact(stream, 8)
            calibration = float(_unpack(stream, "f"))
            names.append(label)
            factors[channel] = calibration * sensitivity / 204.8
    except EOFError as exc:
        raise ValueError("Neuroscan file ended inside an electrode header") from exc
    return names, baselines, factors


def _read_sweep_header(stream: Any) -> _SweepHeader | None:
    raw = stream.read(13)
    if not raw:
        return None
    if len(raw) != 13:
        return None
    accept, event_type, _correct, reaction_time, response, _reserved = struct.unpack("<BHHfHH", raw)
    return _SweepHeader(accept, event_type, float(reaction_time), response)


def _read_exact(stream: Any, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise EOFError
    return value


def _unpack(stream: Any, code: str) -> Any:
    size = struct.calcsize(code)
    return struct.unpack(f"<{code}", _read_exact(stream, size))[0]


def _one_based_indices(value: Any, size: int, name: str) -> np.ndarray:
    if _is_all(value):
        return np.arange(size, dtype=int)
    numeric = np.asarray(value, dtype=float).reshape(-1)
    if np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} indices must be integers")
    indices = numeric.astype(int) - 1
    if np.any(indices < 0) or np.any(indices >= size):
        raise ValueError(f"{name} indices must be 1-based and within the file")
    return indices


def _selection_set(value: Any, size: int, name: str) -> set[int] | None:
    if _is_all(value):
        return None
    return set((_one_based_indices(value, size, name) + 1).tolist())


def _is_all(value: Any) -> bool:
    return isinstance(value, str) and value.lower() == "all"


def _selected_sweep(
    sweep: int,
    metadata: _SweepHeader,
    *,
    selected_trials: set[int] | None,
    typerange: Any,
    accepttype: Any,
    rtrange: Any,
    responsetype: Any,
) -> bool:
    return (
        (selected_trials is None or sweep in selected_trials)
        and _member(metadata.type, typerange)
        and _member(metadata.accept, accepttype)
        and _reaction_time_matches(metadata.reaction_time, rtrange)
        and _member(metadata.response, responsetype)
    )


def _member(value: float, selection: Any) -> bool:
    if _is_all(selection):
        return True
    return bool(np.any(np.asarray(selection, dtype=float).reshape(-1) == value))


def _reaction_time_matches(value: float, selection: Any) -> bool:
    if _is_all(selection):
        return True
    limits = np.asarray(selection, dtype=float).reshape(-1)
    if limits.size == 2:
        return bool(limits[0] <= value <= limits[1])
    return bool(np.any(limits == value))


__all__ = ["loadeeg"]

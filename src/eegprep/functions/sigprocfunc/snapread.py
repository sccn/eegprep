"""Read SnapMaster standard binary data files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def snapread(
    filename: str | Path,
    seekframes: int = 0,
    *,
    event_channel: int = 1,
    event_thresh: float = 2.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Read a SnapMaster ``.SMA`` file into data, params, events, and header."""
    with Path(filename).open("rb") as stream:
        header, nchan, nframes, srate = _read_header(stream)
        marker = stream.read(1)
        if marker not in {b"", b"\xaa"}:
            stream.seek(-1, 1)
        if seekframes:
            stream.seek(int(seekframes) * int(nchan) * 4, 1)
        count = int(nchan) * (int(nframes) - int(seekframes))
        values = np.fromfile(stream, dtype="<f4", count=count)
    if values.size != count:
        raise ValueError("SnapMaster data ended before the number of header frames")
    data = values.reshape((int(nchan), int(nframes) - int(seekframes)), order="F")
    events = _event_transitions(data[event_channel - 1], event_thresh)
    data = np.delete(data, event_channel - 1, axis=0)
    params = np.asarray([data.shape[0], data.shape[1], float(srate)], dtype=float)
    return data, params, events, header


def _read_header(stream: Any) -> tuple[str, int, int, float]:
    header_lines = []
    nchan = nframes = None
    srate = None
    while True:
        raw = stream.readline()
        if raw == b"":
            raise ValueError("SnapMaster header ended before the TR marker")
        line = raw.decode("latin-1")
        if line.startswith('"TR"'):
            stream.readline()
            break
        header_lines.append(line)
        if line.startswith('"NCHAN%"'):
            nchan = int(_value_after_equals(line))
        elif line.startswith('"NUM.POINTS"'):
            nframes = int(_value_after_equals(line))
        elif line.startswith('"ACT.FREQ"'):
            srate = float(_value_after_equals(line))
    if nchan is None or nframes is None or srate is None:
        raise ValueError("SnapMaster header is missing NCHAN%, NUM.POINTS, or ACT.FREQ")
    return "".join(header_lines), nchan, nframes, srate


def _value_after_equals(line: str) -> str:
    return line.split("=", 1)[1].strip()


def _event_transitions(channel: np.ndarray, threshold: float) -> np.ndarray:
    events = np.zeros(channel.shape[0], dtype=int)
    previous = np.abs(channel[:-1]) < float(threshold)
    current = np.abs(channel[1:]) > float(threshold)
    events[1:][previous & current] = 1
    return events


__all__ = ["snapread"]

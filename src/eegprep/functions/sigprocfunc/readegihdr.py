"""Read headers from EGI Simple Binary RAW files."""

from __future__ import annotations

from datetime import datetime
from numbers import Integral
from pathlib import Path
import struct
from typing import BinaryIO, Any

import numpy as np


_FIXED_HEADER = struct.Struct(">i6hi5h")
_CONTINUOUS_VERSIONS = {2, 4, 6}
_SEGMENTED_VERSIONS = {3, 5, 7}
_SAMPLE_DTYPES = {2: ">i2", 3: ">i2", 4: ">f4", 5: ">f4", 6: ">f8", 7: ">f8"}


def readegihdr(filename: str | Path, forceversion: int | None = None) -> dict[str, Any]:
    """Read an EGI Simple Binary header.

    Args:
        filename: EGI ``.RAW`` file.
        forceversion: Optional version from 2 through 7 to use instead of the
            version stored in the file. This is useful for files whose header
            reports the wrong member of the same continuous or segmented
            format family.

    Returns:
        Header fields corresponding to EEGLAB's ``readegihdr`` structure,
        plus explicit recording time, sample dtype, and header byte count.
    """
    with Path(filename).open("rb") as stream:
        return _read_egi_header(stream, forceversion)


def _read_egi_header(stream: BinaryIO, forceversion: int | None = None) -> dict[str, Any]:
    values = _FIXED_HEADER.unpack(_read_exact(stream, _FIXED_HEADER.size, "fixed EGI header"))
    (
        file_version,
        year,
        month,
        day,
        hour,
        minute,
        second,
        millisecond,
        samp_rate,
        nchan,
        gain,
        bits,
        signal_range,
    ) = values
    version = file_version if forceversion is None else _validated_version(forceversion)
    if version not in _SAMPLE_DTYPES:
        raise ValueError("EGI Simple Binary versions 2 through 7 are supported")
    if samp_rate <= 0 or nchan <= 0:
        raise ValueError("EGI header must contain a positive sample rate and channel count")

    samples = 0
    segments = 0
    segsamps = 0
    catname: list[str] = []
    if version in _CONTINUOUS_VERSIONS:
        samples = _read_scalar(stream, ">i", "sample count")
        if samples < 0:
            raise ValueError("EGI sample count must not be negative")
    elif version in _SEGMENTED_VERSIONS:
        categories = _read_scalar(stream, ">h", "category count")
        if categories < 0:
            raise ValueError("EGI category count must not be negative")
        for _ in range(categories):
            name_length = _read_scalar(stream, ">B", "category-name length")
            catname.append(_decode_text(_read_exact(stream, name_length, "category name")))
        segments = _read_scalar(stream, ">h", "segment count")
        segsamps = _read_scalar(stream, ">i", "samples per segment")
        if segments < 0 or segsamps < 0:
            raise ValueError("EGI segment counts must not be negative")

    eventtypes = _read_scalar(stream, ">h", "event-type count")
    if eventtypes < 0:
        raise ValueError("EGI event-type count must not be negative")
    eventcode = [_decode_text(_read_exact(stream, 4, "event code")).rstrip() for _ in range(eventtypes)]

    return {
        "version": version,
        "file_version": file_version,
        "samp_rate": samp_rate,
        "nchan": nchan,
        "gain": gain,
        "bits": bits,
        "range": signal_range,
        "samples": samples,
        "segments": segments,
        "segsamps": segsamps,
        "eventtypes": eventtypes,
        "categories": len(catname),
        "catname": catname,
        "eventcode": eventcode,
        "recording_time": datetime(year, month, day, hour, minute, second, millisecond * 1000),
        "segmented": version in _SEGMENTED_VERSIONS,
        "sample_dtype": _SAMPLE_DTYPES[version],
        "sample_width": np.dtype(_SAMPLE_DTYPES[version]).itemsize,
        "header_bytes": stream.tell(),
    }


def _validated_version(version: int) -> int:
    if isinstance(version, bool) or not isinstance(version, Integral) or version not in _SAMPLE_DTYPES:
        raise ValueError("forceversion must be an integer from 2 through 7")
    return int(version)


def _read_scalar(stream: BinaryIO, format_string: str, field: str) -> int:
    parser = struct.Struct(format_string)
    return int(parser.unpack(_read_exact(stream, parser.size, field))[0])


def _read_exact(stream: BinaryIO, size: int, field: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of file while reading {field}")
    return data


def _decode_text(data: bytes) -> str:
    return data.decode("latin-1").rstrip("\x00")


__all__ = ["readegihdr"]

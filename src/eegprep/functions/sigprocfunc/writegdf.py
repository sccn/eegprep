"""Write continuous EEG data in General Data Format 1.25."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import struct
from typing import Any
import warnings

import numpy as np

_GDF_VERSION = b"GDF 1.25"
_FIXED_HEADER_BYTES = 256
_SIGNAL_HEADER_BYTES = 256
_FLOAT64_GDF_TYPE = 17
_MAX_UINT16 = 2**16 - 1
_MAX_UINT24 = 2**24 - 1
_MAX_UINT32 = 2**32 - 1
_PRIVATE_EVENT_CODE_START = 0x8000
_BOUNDARY_EVENT_CODE = 0x7FFE


def writegdf(
    filename: str | Path,
    data: Any,
    sampling_rate: float,
    *,
    labels: list[str] | tuple[str, ...] | None = None,
    events: Any = None,
    subject: str = "",
    recording: str = "",
) -> dict[str, int]:
    """Write finite continuous channel-major data as GDF 1.25.

    Floating-point GDF samples preserve the input values without integer
    quantization. Numeric event types retain their codes. Non-numeric event
    labels receive deterministic file-local codes in the private range and are
    returned so callers can record that mapping.

    Args:
        filename: Destination ending in ``.gdf``.
        data: Channels by samples matrix.
        sampling_rate: Samples per second.
        labels: Optional channel labels; defaults to ``Ch1``, ``Ch2``, etc.
        events: Optional event dictionaries with 1-based sample ``latency``.
        subject: Optional subject identifier.
        recording: Optional recording identifier.

    Returns:
        Mapping from non-numeric event labels to their stored uint16 codes.
    """
    path = Path(filename).expanduser()
    if path.suffix.lower() != ".gdf":
        raise ValueError("writegdf output must end in .gdf")
    matrix, rate, channel_names = _validated_data(data, sampling_rate, labels)
    record_duration = _record_duration(matrix.shape[1], rate)
    digital_minimum, digital_maximum = _channel_limits(matrix)
    encoded_events, label_codes = _encoded_events(events, matrix.shape[1], matrix.shape[0])
    fixed = _fixed_header(subject, recording, matrix.shape[0], record_duration)
    signal = _signal_header(channel_names, matrix.shape[1], digital_minimum, digital_maximum)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(fixed)
        stream.write(signal)
        stream.write(np.asarray(matrix, dtype="<f8", order="C").tobytes(order="C"))
        stream.write(_event_table(encoded_events, rate))
    return label_codes


def _validated_data(
    data: Any,
    sampling_rate: float,
    labels: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, float, list[str]]:
    raw = np.asarray(data)
    if not np.issubdtype(raw.dtype, np.number):
        raise TypeError("GDF data must be numeric")
    if np.iscomplexobj(raw):
        raise ValueError("GDF data must be real")
    data = raw.astype(float, copy=False)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("writegdf requires non-empty continuous 2-D channel-by-sample data")
    if not np.all(np.isfinite(data)):
        raise ValueError("writegdf requires finite EEG data")
    rate = float(sampling_rate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sampling_rate must be finite and positive")
    channel_names = [f"Ch{index + 1}" for index in range(data.shape[0])] if labels is None else list(labels)
    if len(channel_names) != data.shape[0]:
        raise ValueError("channel labels do not match the data rows")
    for label in channel_names:
        _fixed_text(label, 16, "channel label")
    return data, rate, channel_names


def _record_duration(samples: int, sampling_rate: float) -> Fraction:
    duration = Fraction(samples, 1) / Fraction(str(sampling_rate))
    duration = duration.limit_denominator(_MAX_UINT32)
    if duration.numerator > _MAX_UINT32 or duration.denominator > _MAX_UINT32:
        raise ValueError("GDF record duration cannot represent this sample count and sampling rate")
    represented_rate = samples * duration.denominator / duration.numerator
    if not np.isclose(represented_rate, sampling_rate, rtol=1e-12, atol=0):
        raise ValueError("GDF record duration cannot represent the sampling rate accurately")
    return duration


def _channel_limits(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.floor(np.min(data, axis=1))
    upper = np.ceil(np.max(data, axis=1))
    constant = lower == upper
    lower[constant] -= 1
    upper[constant] += 1
    info = np.iinfo(np.int64)
    if np.any(lower < info.min) or np.any(upper > info.max):
        raise ValueError("GDF channel range exceeds the format's int64 calibration limits")
    return lower.astype("<i8"), upper.astype("<i8")


def _fixed_header(
    subject: str,
    recording: str,
    channels: int,
    record_duration: Fraction,
) -> bytes:
    patient = _fixed_text(str(subject), 80, "subject")
    recording_field = _fixed_text(str(recording), 80, "recording")
    header_bytes = _FIXED_HEADER_BYTES + channels * _SIGNAL_HEADER_BYTES
    return b"".join(
        (
            _GDF_VERSION,
            patient,
            recording_field,
            b"0" * 16,
            struct.pack("<q", header_bytes),
            bytes(44),
            struct.pack("<q", 1),
            struct.pack("<II", record_duration.numerator, record_duration.denominator),
            struct.pack("<I", channels),
        )
    )


def _signal_header(
    labels: list[str],
    samples: int,
    digital_minimum: np.ndarray,
    digital_maximum: np.ndarray,
) -> bytes:
    channels = len(labels)
    if samples > np.iinfo(np.int32).max:
        raise ValueError("GDF data exceed the per-record sample limit")
    return b"".join(
        (
            b"".join(_fixed_text(label, 16, "channel label") for label in labels),
            bytes(80 * channels),
            b"".join(_fixed_text("uV", 8, "physical dimension") for _ in labels),
            digital_minimum.astype("<f8").tobytes(),
            digital_maximum.astype("<f8").tobytes(),
            digital_minimum.astype("<i8").tobytes(),
            digital_maximum.astype("<i8").tobytes(),
            bytes(80 * channels),
            np.full(channels, samples, dtype="<i4").tobytes(),
            np.full(channels, _FLOAT64_GDF_TYPE, dtype="<i4").tobytes(),
            bytes(32 * channels),
        )
    )


def _encoded_events(
    raw_events: Any,
    samples: int,
    channels: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, int]]:
    records = _event_records(raw_events)
    label_codes: dict[str, int] = {}
    encoded = []
    reserved_codes = {code for event in records if (code := _standard_event_code(event.get("type", 0))) is not None}
    next_private_code = _PRIVATE_EVENT_CODE_START
    for event in records:
        latency = _finite_value(event.get("latency", 1), "event latency")
        position = int(np.floor(latency + 0.5))
        if position < 1 or position > samples:
            raise ValueError(f"event latency {latency} is outside the 1..{samples} sample range")
        duration = _finite_value(event.get("duration", 0) or 0, "event duration")
        if duration < 0:
            raise ValueError("event duration must be non-negative")
        stored_duration = int(np.floor(duration + 0.5))
        if stored_duration > _MAX_UINT32:
            raise ValueError("event duration exceeds the GDF uint32 limit")
        channel = _event_channel(event.get("channel", 0), channels)
        code, next_private_code = _event_code(
            event.get("type", 0),
            label_codes,
            reserved_codes,
            next_private_code,
        )
        encoded.append((position, code, channel, stored_duration))
    if label_codes:
        warnings.warn(
            "GDF stores uint16 event codes rather than free-text labels; non-numeric labels were assigned "
            f"file-local codes: {label_codes}",
            RuntimeWarning,
            stacklevel=3,
        )
    return encoded, label_codes


def _event_records(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, dict):
        return [dict(events)]
    return [dict(event) for event in events]


def _event_code(
    value: Any,
    label_codes: dict[str, int],
    reserved_codes: set[int],
    next_private: int,
) -> tuple[int, int]:
    standard = _standard_event_code(value)
    if standard is not None:
        return standard, next_private
    text = str(value).strip()
    if text not in label_codes:
        while next_private in reserved_codes or next_private in label_codes.values():
            next_private += 1
        if next_private > _MAX_UINT16:
            raise ValueError("too many distinct non-numeric GDF event labels")
        label_codes[text] = next_private
        next_private += 1
    return label_codes[text], next_private


def _standard_event_code(value: Any) -> int | None:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "boundary":
            return _BOUNDARY_EVENT_CODE
        try:
            numeric = float(text)
        except ValueError:
            return None
    else:
        numeric = _finite_value(value, "event type")
    if numeric != np.trunc(numeric) or not 0 <= numeric <= _MAX_UINT16:
        raise ValueError("numeric GDF event types must be integers from 0 through 65535")
    return int(numeric)


def _event_channel(value: Any, channels: int) -> int:
    numeric = _finite_value(value, "event channel")
    if numeric != np.trunc(numeric) or not 0 <= numeric <= channels:
        raise ValueError(f"event channel must be an integer from 0 through {channels}")
    return int(numeric)


def _event_table(events: list[tuple[int, int, int, int]], sampling_rate: float) -> bytes:
    if events and sampling_rate != np.trunc(sampling_rate):
        raise ValueError("GDF 1.25 event tables require an integer sampling rate for exact event timing")
    event_rate = int(sampling_rate) if events else max(1, min(_MAX_UINT24, round(sampling_rate)))
    if not 1 <= event_rate <= _MAX_UINT24:
        raise ValueError("GDF event sampling rate must round to a uint24 positive integer")
    prefix = b"\x03" + event_rate.to_bytes(3, "little") + struct.pack("<I", len(events))
    if not events:
        return prefix
    positions, types, channels, durations = (np.asarray(values) for values in zip(*events, strict=True))
    return b"".join(
        (
            prefix,
            positions.astype("<u4").tobytes(),
            types.astype("<u2").tobytes(),
            channels.astype("<u2").tobytes(),
            durations.astype("<u4").tobytes(),
        )
    )


def _fixed_text(value: str, width: int, name: str) -> bytes:
    try:
        encoded = value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError(f"GDF {name} must use Latin-1 characters") from exc
    if len(encoded) > width:
        raise ValueError(f"GDF {name} must fit in {width} bytes")
    return encoded.ljust(width)


def _finite_value(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


__all__ = ["writegdf"]

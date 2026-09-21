"""Open BioSemi Data Format files for record-oriented access."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


logger = logging.getLogger(__name__)

_FIXED_HEADER_SIZE = 256
_SIGNAL_HEADER_SIZE = 256
_BDF_BYTES_PER_SAMPLE = 3


def openbdf(filename: str | Path, *, return_header: bool = False) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Read a BDF header and prepare an EEGLAB-compatible record dataset.

    Unlike the historical MATLAB helper, this function does not leave an open
    file descriptor in the returned structure. ``readbdf`` reopens the stored
    path for each read, so the result is safe to keep and serialize.

    Args:
        filename: BioSemi Data Format file.
        return_header: Also return the raw 256-byte fixed header as a Latin-1
            string.

    Returns:
        A dictionary containing ``Head`` and ``MX``. If ``return_header`` is
        true, returns ``(dataset, raw_header)``.
    """
    path = Path(filename).expanduser().resolve()
    with path.open("rb") as stream:
        fixed = _read_exact(stream, _FIXED_HEADER_SIZE, "fixed BDF header")
        version = fixed[:8].decode("latin-1")
        if version != "\xffBIOSEMI":
            raise ValueError(f"{path} is not a BioSemi BDF file")

        head_length = _parse_integer(fixed[184:192], "header length")
        record_count = _parse_integer(fixed[236:244], "record count")
        duration = _parse_float(fixed[244:252], "record duration")
        signal_count = _parse_integer(fixed[252:256], "signal count")
        if signal_count <= 0:
            raise ValueError("BDF signal count must be positive")
        expected_head_length = _FIXED_HEADER_SIZE + _SIGNAL_HEADER_SIZE * signal_count
        if head_length != expected_head_length:
            raise ValueError(
                f"BDF header length is {head_length}, expected {expected_head_length} for {signal_count} signals"
            )
        if duration <= 0:
            raise ValueError("BDF record duration must be positive")

        labels = _read_text_fields(stream, 16, signal_count, "signal labels")
        transducers = _read_text_fields(stream, 80, signal_count, "transducers")
        physical_dimensions = _read_text_fields(stream, 8, signal_count, "physical dimensions")
        physical_minimum = _read_numeric_fields(stream, 8, signal_count, "physical minimum")
        physical_maximum = _read_numeric_fields(stream, 8, signal_count, "physical maximum")
        digital_minimum = _read_numeric_fields(stream, 8, signal_count, "digital minimum")
        digital_maximum = _read_numeric_fields(stream, 8, signal_count, "digital maximum")
        prefiltering = _read_text_fields(stream, 80, signal_count, "prefiltering")
        samples_per_record = _read_numeric_fields(stream, 8, signal_count, "samples per record")
        _read_exact(stream, 32 * signal_count, "per-signal reserved header")

    digital_minimum, digital_maximum = _valid_digital_limits(digital_minimum, digital_maximum, signal_count)
    physical_minimum, physical_maximum = _valid_physical_limits(
        physical_minimum,
        physical_maximum,
        digital_minimum,
        digital_maximum,
        signal_count,
    )
    samples_per_record = _valid_samples_per_record(samples_per_record, signal_count)
    samples_in_block = int(samples_per_record.sum())
    record_bytes = samples_in_block * _BDF_BYTES_PER_SAMPLE
    if record_count == -1:
        record_count = _infer_record_count(path, head_length, record_bytes)
    elif record_count < 0:
        raise ValueError("BDF record count must be nonnegative or -1 for unknown")

    calibration, offset = _calibration(
        physical_minimum,
        physical_maximum,
        digital_minimum,
        digital_maximum,
    )
    maximum_samples = int(samples_per_record.max())
    channel_selection = samples_per_record == maximum_samples
    channel_types = "".join(
        _channel_type(label, selected) for label, selected in zip(labels, channel_selection, strict=True)
    )
    matlab_linear_indices = np.concatenate(
        [
            channel * maximum_samples + np.arange(1, count + 1, dtype=np.intp)
            for channel, count in enumerate(samples_per_record)
        ]
    )
    start_date = fixed[168:176].decode("ascii", errors="replace")
    start_time = fixed[176:184].decode("ascii", errors="replace")
    time_zero, start_datetime = _parse_start(start_date, start_time, version)

    head: dict[str, Any] = {
        "FILE": {
            "FID": None,
            "OPEN": 1,
            "Ext": path.suffix.removeprefix("."),
            "Name": path.stem,
            "Path": str(path.parent),
            "POS": head_length,
        },
        "FileName": str(path),
        "VERSION": version,
        "PID": fixed[8:88].decode("ascii", errors="replace").rstrip(),
        "RID": fixed[88:168].decode("ascii", errors="replace").rstrip(),
        "T0": time_zero,
        "StartDateTime": start_datetime,
        "HeadLen": head_length,
        "NRec": record_count,
        "Dur": duration,
        "NS": signal_count,
        "Label": labels,
        "Transducer": transducers,
        "PhysDim": physical_dimensions,
        "PhysMin": physical_minimum,
        "PhysMax": physical_maximum,
        "DigMin": digital_minimum,
        "DigMax": digital_maximum,
        "PreFilt": prefiltering,
        "SPR": samples_per_record,
        "Cal": calibration,
        "Off": offset,
        "Calib": np.vstack((offset, np.diag(calibration))),
        "SampleRate": samples_per_record.astype(float) / duration,
        "Chan_Select": channel_selection,
        "ChanTyp": channel_types,
        "AS": {"spb": samples_in_block, "IDX2": matlab_linear_indices},
    }
    dataset = {"Head": head, "MX": {"ReRef": 1}}
    if return_header:
        return dataset, fixed.decode("latin-1")
    return dataset


def _read_exact(stream: BinaryIO, size: int, description: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"file ended before the complete {description}")
    return data


def _read_text_fields(stream: BinaryIO, width: int, count: int, description: str) -> list[str]:
    raw = _read_exact(stream, width * count, description)
    return [
        raw[index * width : (index + 1) * width].decode("ascii", errors="replace").rstrip() for index in range(count)
    ]


def _read_numeric_fields(
    stream: BinaryIO,
    width: int,
    count: int,
    description: str,
) -> np.ndarray | None:
    fields = _read_text_fields(stream, width, count, description)
    try:
        return np.asarray([float(field) for field in fields], dtype=float)
    except ValueError:
        return None


def _parse_integer(raw: bytes, description: str) -> int:
    value = _parse_float(raw, description)
    if value != np.trunc(value):
        raise ValueError(f"BDF {description} must be an integer")
    return int(value)


def _parse_float(raw: bytes, description: str) -> float:
    try:
        value = float(raw.decode("ascii").strip())
    except ValueError as exc:
        raise ValueError(f"invalid BDF {description}") from exc
    if not np.isfinite(value):
        raise ValueError(f"BDF {description} must be finite")
    return value


def _valid_digital_limits(
    minimum: np.ndarray | None,
    maximum: np.ndarray | None,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if minimum is None or minimum.size != count:
        logger.warning("BDF digital minimum is missing or invalid; using int16 limits")
        minimum = np.full(count, -32768.0)
    if maximum is None or maximum.size != count:
        logger.warning("BDF digital maximum is missing or invalid; using int16 limits")
        maximum = np.full(count, 32767.0)
    if np.any(minimum >= maximum):
        logger.warning("BDF digital minimum is not smaller than maximum")
    return minimum, maximum


def _valid_physical_limits(
    minimum: np.ndarray | None,
    maximum: np.ndarray | None,
    digital_minimum: np.ndarray,
    digital_maximum: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if minimum is None or minimum.size != count:
        logger.warning("BDF physical minimum is missing or invalid; using digital minimum")
        minimum = digital_minimum.copy()
    if maximum is None or maximum.size != count:
        logger.warning("BDF physical maximum is missing or invalid; using digital maximum")
        maximum = digital_maximum.copy()
    if np.any(minimum >= maximum):
        logger.warning("BDF physical minimum is not smaller than maximum; using digital limits")
        return digital_minimum.copy(), digital_maximum.copy()
    return minimum, maximum


def _valid_samples_per_record(values: np.ndarray | None, count: int) -> np.ndarray:
    if values is None or values.size != count:
        raise ValueError("BDF samples per record are missing or invalid")
    if np.any(values <= 0) or np.any(values != np.trunc(values)):
        raise ValueError("BDF samples per record must contain positive integers")
    return values.astype(np.intp)


def _calibration(
    physical_minimum: np.ndarray,
    physical_maximum: np.ndarray,
    digital_minimum: np.ndarray,
    digital_maximum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = digital_minimum < digital_maximum
    calibration = np.ones(digital_minimum.shape, dtype=float)
    offset = np.zeros(digital_minimum.shape, dtype=float)
    calibration[valid] = (physical_maximum[valid] - physical_minimum[valid]) / (
        digital_maximum[valid] - digital_minimum[valid]
    )
    offset[valid] = physical_minimum[valid] - calibration[valid] * digital_minimum[valid]
    positive = calibration > 0
    calibration[~positive] = 1.0
    offset[~positive] = 0.0
    return calibration, offset


def _infer_record_count(path: Path, head_length: int, record_bytes: int) -> int:
    data_bytes = path.stat().st_size - head_length
    if data_bytes < 0:
        raise ValueError("BDF file is shorter than its declared header")
    count, remainder = divmod(data_bytes, record_bytes)
    if remainder:
        logger.warning("BDF file has %d trailing bytes outside complete records", remainder)
    return count


def _channel_type(label: str, selected: np.bool_, /) -> str:
    upper = label.upper()
    for token, channel_type in (("ECG", "C"), ("EKG", "C"), ("EEG", "E"), ("EOG", "O"), ("EMG", "M")):
        if token in upper:
            return channel_type
    return "N" if selected else " "


def _parse_start(date: str, time: str, version: str) -> tuple[list[int], datetime | None]:
    try:
        day, month, short_year = (int(value) for value in date.split("."))
        hour, minute, second = (int(value) for value in time.split("."))
    except ValueError:
        logger.warning("BDF start date or time is invalid")
        return [0, 0, 0, 0, 0, 0], None
    matlab_year = short_year
    if version.startswith("0"):
        matlab_year = (2000 if short_year < 91 else 1900) + short_year
    full_year = (2000 if short_year < 85 else 1900) + short_year
    try:
        parsed = datetime(full_year, month, day, hour, minute, second)
    except ValueError:
        logger.warning("BDF start date or time is outside the calendar")
        parsed = None
    return [matlab_year, month, day, hour, minute, second], parsed


__all__ = ["openbdf"]

"""Load BrainVision Data Exchange recordings into EEGPrep datasets."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import warnings

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.popfunc._pop_utils import format_history_value
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


_HEADER_PREFIX = "Brain Vision Data Exchange Header File"
_MARKER_PREFIX = "Brain Vision Data Exchange Marker File"
_BINARY_DTYPES = {
    "INT_16": np.dtype("<i2"),
    "UINT_16": np.dtype("<u2"),
    "IEEE_FLOAT_32": np.dtype("<f4"),
}
_VOLTAGE_TO_MICROVOLTS = {
    "v": 1_000_000.0,
    "mv": 1_000.0,
    "uv": 1.0,
    "µv": 1.0,
    "μv": 1.0,
    "nv": 0.001,
}


def pop_loadbv(
    path: str | Path,
    hdrfile: str | Path | None = None,
    srange: Any = None,
    chans: Any = None,
    metadata: bool = False,
    *,
    return_com: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Load a BrainVision ``.vhdr`` recording.

    Args:
        path: Directory containing the recording, or the full header path.
        hdrfile: Header filename when ``path`` is a directory.
        srange: One-based first sample, or inclusive ``[first, last]`` range.
        chans: One-based channel indices to load, in the requested order.
        metadata: Read metadata and events without loading signal samples.
        return_com: Return ``(EEG, command)`` for command-history workflows.

    Returns:
        An EEGPrep EEG dictionary, optionally paired with its replay command.

    Notes:
        Voltage channels are converted to EEGPrep's microvolt convention. The
        original BrainVision unit and resolution remain available as
        ``chanlocs[*]["bvunit"]`` and ``chanlocs[*]["bvresolution"]``.
    """
    header_path = _header_path(path, hdrfile)
    header = _read_configuration(header_path, expected_prefix=_HEADER_PREFIX)
    common = _section(header, "common infos")
    channel_count = _required_positive_int(common, "NumberOfChannels")
    sampling_interval = _required_positive_float(common, "SamplingInterval")
    srate = 1_000_000.0 / sampling_interval
    data_type = common.get("datatype", "TIMEDOMAIN").strip().upper()
    if data_type != "TIMEDOMAIN":
        raise ValueError(f"Unsupported BrainVision data type: {data_type}")
    data_format = _required(common, "DataFormat").upper()
    orientation = _required(common, "DataOrientation").upper()
    if orientation not in {"MULTIPLEXED", "VECTORIZED"}:
        raise ValueError(f"Unsupported BrainVision data orientation: {orientation}")

    data_file = _required(common, "DataFile")
    data_path = _referenced_file(header_path, data_file, fallback_suffixes=(".eeg", ".dat"))
    if data_format == "BINARY":
        dtype = _binary_dtype(header)
        point_count = _binary_point_count(data_path, channel_count, dtype, common)
    elif data_format == "ASCII":
        dtype = None
        ascii_data = _read_ascii_data(data_path, header, channel_count, orientation)
        point_count = int(ascii_data.shape[1])
        _validate_declared_points(common, point_count)
    else:
        raise ValueError(f"Unsupported BrainVision data format: {data_format}")

    first, stop = _sample_bounds(srange, point_count)
    selected_channels = _channel_indices(chans, channel_count)
    chanlocs, scales = _channel_metadata(header, channel_count, selected_channels)
    if metadata:
        data = np.array([], dtype=float)
    elif data_format == "BINARY":
        data = _read_binary_data(
            data_path,
            dtype,
            channel_count,
            point_count,
            orientation,
            selected_channels,
            first,
            stop,
        )
        data *= scales[:, np.newaxis]
    else:
        data = np.asarray(ascii_data[np.ix_(selected_channels, np.arange(first, stop))], dtype=float)
        data *= scales[:, np.newaxis]

    events = _read_events(
        header_path,
        common,
        channel_count=channel_count,
        selected_channels=selected_channels,
        first=first,
        stop=stop,
    )
    command = _history_command(header_path, srange, chans, metadata)
    eeg = _build_eeg(
        header_path,
        data_file=data_file,
        data=data,
        metadata=metadata,
        chanlocs=chanlocs,
        events=events,
        point_count=stop - first,
        srate=srate,
        common=common,
        command=command,
        selected_channels=selected_channels,
        data_format=data_format,
        orientation=orientation,
    )
    return (eeg, command) if return_com else eeg


def _header_path(path: str | Path, hdrfile: str | Path | None) -> Path:
    base = Path(path).expanduser()
    if hdrfile is None:
        candidate = base
    else:
        candidate = Path(hdrfile).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
    if candidate.suffix.lower() in {".eeg", ".dat"}:
        candidate = candidate.with_suffix(".vhdr")
    if candidate.suffix.lower() != ".vhdr":
        raise ValueError("pop_loadbv requires a BrainVision .vhdr header")
    resolved = _case_insensitive_file(candidate)
    if resolved is None:
        raise FileNotFoundError(f"BrainVision header file not found: {candidate}")
    return resolved


def _read_configuration(path: Path, *, expected_prefix: str) -> dict[str, dict[str, str]]:
    text = _decode_brainvision_text(path)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first_line.casefold().startswith(expected_prefix.casefold()):
        raise ValueError(f"Not a supported {expected_prefix}: {path}")
    configuration: dict[str, dict[str, str]] = {}
    current_section: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        section_match = re.fullmatch(r"\[([^]]+)]", line)
        if section_match is not None:
            current_section = section_match.group(1).strip().casefold()
            configuration.setdefault(current_section, {})
            continue
        if current_section is None:
            continue
        if current_section in {"comment", "marker infos"}:
            continue
        if "=" not in line:
            raise ValueError(f"Malformed BrainVision configuration {path} at line {line_number}: expected key=value")
        key, value = line.split("=", 1)
        normalized_key = key.strip().casefold()
        if not normalized_key:
            raise ValueError(f"Malformed BrainVision configuration {path} at line {line_number}: empty key")
        section = configuration[current_section]
        if normalized_key in section:
            raise ValueError(
                f"Malformed BrainVision configuration {path} at line {line_number}: duplicate {key.strip()}"
            )
        section[normalized_key] = value.strip()
    if not configuration:
        raise ValueError(f"BrainVision file contains no configuration sections: {path}")
    return configuration


def _decode_brainvision_text(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"BrainVision text file is neither UTF-8 nor Windows-1252: {path}")


def _section(configuration: dict[str, dict[str, str]], name: str) -> dict[str, str]:
    try:
        return configuration[name.casefold()]
    except KeyError as error:
        raise ValueError(f"BrainVision configuration is missing [{name.title()}]") from error


def _required(section: dict[str, str], field: str) -> str:
    value = section.get(field.casefold(), "").strip()
    if not value:
        raise ValueError(f"BrainVision configuration field {field} is required")
    return value


def _required_positive_int(section: dict[str, str], field: str) -> int:
    value = _required(section, field)
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"BrainVision {field} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"BrainVision {field} must be a positive integer")
    return parsed


def _required_positive_float(section: dict[str, str], field: str) -> float:
    value = _required(section, field)
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"BrainVision {field} must be a positive number") from error
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"BrainVision {field} must be a positive number")
    return parsed


def _binary_dtype(header: dict[str, dict[str, str]]) -> np.dtype:
    binary = _section(header, "binary infos")
    binary_format = _required(binary, "BinaryFormat").upper()
    try:
        dtype = _BINARY_DTYPES[binary_format]
    except KeyError as error:
        raise ValueError(f"Unsupported BrainVision binary format: {binary_format}") from error
    big_endian = binary.get("usebigendianorder", "NO").strip().upper()
    if big_endian not in {"YES", "NO"}:
        raise ValueError("BrainVision UseBigEndianOrder must be YES or NO")
    return dtype.newbyteorder(">" if big_endian == "YES" else "<")


def _binary_point_count(
    data_path: Path,
    channel_count: int,
    dtype: np.dtype,
    common: dict[str, str],
) -> int:
    frame_bytes = channel_count * dtype.itemsize
    byte_count = data_path.stat().st_size
    if byte_count % frame_bytes:
        raise ValueError("BrainVision binary data is truncated: file size is not divisible by the channel frame size")
    point_count = byte_count // frame_bytes
    if point_count < 1:
        raise ValueError("BrainVision data file contains no samples")
    _validate_declared_points(common, point_count)
    return point_count


def _validate_declared_points(common: dict[str, str], point_count: int) -> None:
    declared = common.get("datapoints", "").strip()
    if not declared:
        return
    try:
        declared_count = int(declared)
    except ValueError as error:
        raise ValueError("BrainVision DataPoints must be a positive integer") from error
    if declared_count < 1:
        raise ValueError("BrainVision DataPoints must be a positive integer")
    if declared_count != point_count:
        raise ValueError(
            f"BrainVision DataPoints declares {declared_count} samples but the data file contains {point_count}"
        )


def _sample_bounds(srange: Any, point_count: int) -> tuple[int, int]:
    values = _integer_vector(srange, "srange")
    if not values:
        return 0, point_count
    if len(values) == 1:
        first, last = values[0], point_count
    elif len(values) == 2:
        first, last = values
    else:
        raise ValueError("srange must be a first sample or an inclusive [first, last] range")
    if first < 1 or last < first or last > point_count:
        raise ValueError(f"srange must be within the available 1:{point_count} samples")
    return first - 1, last


def _channel_indices(chans: Any, channel_count: int) -> list[int]:
    values = _integer_vector(chans, "chans")
    if not values:
        return list(range(channel_count))
    if any(index < 1 or index > channel_count for index in values):
        raise ValueError(f"chans must contain 1-based indices in the range 1:{channel_count}")
    return [index - 1 for index in values]


def _integer_vector(value: Any, name: str) -> list[int]:
    if value is None:
        return []
    array = np.asarray(value)
    if array.size == 0:
        return []
    try:
        numeric = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer values") from error
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError(f"{name} must contain integer values")
    return numeric.astype(int).tolist()


def _channel_metadata(
    header: dict[str, dict[str, str]],
    channel_count: int,
    selected_channels: list[int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    channel_infos = _section(header, "channel infos")
    coordinates = header.get("coordinates", {})
    parsed_channels: list[dict[str, Any]] = []
    scales: list[float] = []
    for source_index in selected_channels:
        key = f"ch{source_index + 1}"
        if key not in channel_infos:
            raise ValueError(f"BrainVision [Channel Infos] is missing Ch{source_index + 1}")
        fields = _split_brainvision_fields(channel_infos[key])
        label = fields[0].strip() if fields and fields[0].strip() else f"Ch{source_index + 1}"
        reference = fields[1].strip() if len(fields) > 1 else ""
        resolution = _channel_resolution(fields, source_index)
        original_unit = fields[3].strip() if len(fields) > 3 and fields[3].strip() else "µV"
        unit_key = original_unit.replace("\N{MICRO SIGN}", "µ").casefold()
        unit_factor = _VOLTAGE_TO_MICROVOLTS.get(unit_key)
        is_voltage = unit_factor is not None
        scale = resolution * (unit_factor if unit_factor is not None else 1.0)
        chanloc = {
            "labels": label,
            "ref": reference,
            "type": "EEG" if is_voltage else "MISC",
            "unit": "µV" if is_voltage else original_unit,
            "bvunit": original_unit,
            "bvresolution": resolution,
            "urchan": source_index,
        }
        if key in coordinates:
            chanloc.update(_coordinate_fields(coordinates[key], source_index))
        parsed_channels.append(chanloc)
        scales.append(scale)
    return parsed_channels, np.asarray(scales, dtype=float)


def _channel_resolution(fields: list[str], source_index: int) -> float:
    raw = fields[2].strip() if len(fields) > 2 else ""
    if not raw:
        return 1.0
    try:
        resolution = float(raw)
    except ValueError as error:
        raise ValueError(f"BrainVision Ch{source_index + 1} resolution must be numeric") from error
    if not np.isfinite(resolution):
        raise ValueError(f"BrainVision Ch{source_index + 1} resolution must be finite")
    return resolution


def _coordinate_fields(value: str, source_index: int) -> dict[str, float]:
    fields = _split_brainvision_fields(value)
    if len(fields) < 3:
        raise ValueError(f"BrainVision coordinate Ch{source_index + 1} requires radius, theta, phi")
    try:
        radius, theta, phi = (float(field.strip()) for field in fields[:3])
    except ValueError as error:
        raise ValueError(f"BrainVision coordinate Ch{source_index + 1} must be numeric") from error
    if not np.all(np.isfinite([radius, theta, phi])):
        raise ValueError(f"BrainVision coordinate Ch{source_index + 1} must be finite")
    if radius == theta == phi == 0:
        return {
            "X": np.nan,
            "Y": np.nan,
            "Z": np.nan,
            "sph_radius": np.nan,
            "sph_theta": np.nan,
            "sph_phi": np.nan,
        }
    polar = np.deg2rad(theta)
    azimuth = np.deg2rad(phi)
    bv_x = radius * np.sin(polar) * np.cos(azimuth)
    bv_y = radius * np.sin(polar) * np.sin(azimuth)
    return {
        "X": float(bv_y),
        "Y": float(-bv_x),
        "Z": float(radius * np.cos(polar)),
        "sph_radius": radius,
        "sph_theta": phi - 90.0 * np.sign(theta),
        "sph_phi": -abs(theta) + 90.0,
    }


def _read_binary_data(
    data_path: Path,
    dtype: np.dtype,
    channel_count: int,
    point_count: int,
    orientation: str,
    selected_channels: list[int],
    first: int,
    stop: int,
) -> np.ndarray:
    memory = np.memmap(data_path, dtype=dtype, mode="r")
    if orientation == "MULTIPLEXED":
        samples = memory.reshape(point_count, channel_count)
        return np.asarray(samples[first:stop, selected_channels].T, dtype=float).copy()
    channels = memory.reshape(channel_count, point_count)
    return np.asarray(channels[np.ix_(selected_channels, np.arange(first, stop))], dtype=float).copy()


def _read_ascii_data(
    data_path: Path,
    header: dict[str, dict[str, str]],
    channel_count: int,
    orientation: str,
) -> np.ndarray:
    ascii_infos = header.get("ascii infos", {})
    skip_lines = _nonnegative_ascii_int(ascii_infos, "SkipLines")
    skip_columns = _nonnegative_ascii_int(ascii_infos, "SkipColumns")
    decimal_symbol = ascii_infos.get("decimalsymbol", ".").strip() or "."
    rows: list[list[float]] = []
    for raw_line in _decode_brainvision_text(data_path).splitlines()[skip_lines:]:
        line = raw_line.strip()
        if not line:
            continue
        if decimal_symbol != ".":
            line = line.replace(decimal_symbol, ".")
        fields = [field for field in re.split(r"[\s,;]+", line) if field]
        fields = fields[skip_columns:]
        try:
            rows.append([float(field) for field in fields])
        except ValueError as error:
            raise ValueError(f"BrainVision ASCII data contains a non-numeric value: {raw_line!r}") from error
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("BrainVision ASCII data must be a non-empty rectangular numeric table")
    matrix = np.asarray(rows, dtype=float)
    if orientation == "MULTIPLEXED":
        if matrix.shape[1] != channel_count:
            raise ValueError("BrainVision multiplexed ASCII data column count does not match NumberOfChannels")
        return matrix.T
    if matrix.shape[0] != channel_count:
        raise ValueError("BrainVision vectorized ASCII data row count does not match NumberOfChannels")
    return matrix


def _nonnegative_ascii_int(section: dict[str, str], field: str) -> int:
    raw = section.get(field.casefold(), "0").strip() or "0"
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"BrainVision {field} must be a nonnegative integer") from error
    if value < 0:
        raise ValueError(f"BrainVision {field} must be a nonnegative integer")
    return value


def _read_events(
    header_path: Path,
    common: dict[str, str],
    *,
    channel_count: int,
    selected_channels: list[int],
    first: int,
    stop: int,
) -> list[dict[str, Any]]:
    marker_name = common.get("markerfile", "").strip()
    if not marker_name:
        return []
    try:
        marker_path = _referenced_file(header_path, marker_name, fallback_suffixes=(".vmrk",))
    except FileNotFoundError:
        warnings.warn(
            f"BrainVision marker file not found; importing without events: {marker_name}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    marker_text = _decode_brainvision_text(marker_path)
    if marker_text.strip() == "--corrupted--":
        warnings.warn(
            f"BrainVision marker file is marked corrupted; importing without events: {marker_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    marker = _read_configuration(marker_path, expected_prefix=_MARKER_PREFIX)
    marker_common = marker.get("common infos", {})
    marker_data_file = marker_common.get("datafile", "").strip()
    if marker_data_file and marker_data_file.casefold() != _required(common, "DataFile").casefold():
        warnings.warn(
            "BrainVision header and marker files reference different data files",
            RuntimeWarning,
            stacklevel=2,
        )
    parsed: list[dict[str, Any]] = []
    for key, value in _marker_entries(marker_text, marker_path):
        match = re.fullmatch(r"mk(\d+)", key, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Invalid BrainVision marker key: {key}")
        marker_number = int(match.group(1))
        parsed.append(_parse_marker(value, marker_number, channel_count))
    events: list[dict[str, Any]] = []
    for source_event in parsed:
        latency = float(source_event["latency"])
        if latency < first + 1 or latency > stop:
            continue
        event = dict(source_event)
        event["latency"] = latency - first
        source_channel = int(event["channel"])
        if source_channel:
            event["bvchannel"] = source_channel
            try:
                event["channel"] = selected_channels.index(source_channel - 1) + 1
            except ValueError:
                event["channel"] = 0
        if event["type"] == "boundary":
            event["duration"] = np.nan
        event["urevent"] = len(events)
        events.append(event)
    return events


def _parse_marker(value: str, marker_number: int, channel_count: int) -> dict[str, Any]:
    fields = _split_brainvision_fields(value)
    if len(fields) < 5:
        raise ValueError(f"BrainVision marker Mk{marker_number} requires at least five fields")
    marker_type = fields[0].strip()
    description = fields[1].strip()
    try:
        latency = float(fields[2].strip())
        duration_float = float(fields[3].strip())
        channel_float = float(fields[4].strip())
    except ValueError as error:
        raise ValueError(f"BrainVision marker Mk{marker_number} has invalid numeric fields") from error
    if not np.all(np.isfinite([latency, duration_float, channel_float])):
        raise ValueError(f"BrainVision marker Mk{marker_number} has non-finite numeric fields")
    if duration_float != np.floor(duration_float) or channel_float != np.floor(channel_float):
        raise ValueError(f"BrainVision marker Mk{marker_number} duration and channel must be integers")
    duration = int(duration_float)
    channel = int(channel_float)
    if latency < 1 or duration < 0 or channel < 0 or channel > channel_count:
        raise ValueError(f"BrainVision marker Mk{marker_number} has out-of-range numeric fields")
    event: dict[str, Any] = {
        "type": "boundary" if marker_type.casefold() in {"new segment", "dc correction"} else description,
        "latency": latency,
        "duration": duration,
        "channel": channel,
        "code": marker_type,
        "bvmknum": marker_number,
    }
    if len(fields) > 5 and fields[5].strip():
        event["bvtime"] = fields[5].strip()
    if len(fields) > 6 and fields[6].strip():
        event["visible"] = fields[6].strip()
    return event


def _split_brainvision_fields(value: str) -> list[str]:
    placeholder = "\0BRAINVISION_COMMA\0"
    escaped = value.replace(r"\1", placeholder).replace("\x01", placeholder)
    return [field.replace(placeholder, ",") for field in escaped.split(",")]


def _marker_entries(text: str, path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    in_marker_infos = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        section_match = re.fullmatch(r"\[([^]]+)]", line)
        if section_match is not None:
            in_marker_infos = section_match.group(1).strip().casefold() == "marker infos"
            continue
        if not in_marker_infos or not line or line.startswith(";"):
            continue
        if "=" not in line:
            raise ValueError(f"Malformed BrainVision marker file {path} at line {line_number}")
        key, value = line.split("=", 1)
        entries.append((key.strip(), value.strip()))
    return entries


def _referenced_file(header_path: Path, filename: str, *, fallback_suffixes: tuple[str, ...]) -> Path:
    requested = Path(filename)
    candidate = requested if requested.is_absolute() else header_path.parent / requested
    resolved = _case_insensitive_file(candidate)
    if resolved is not None:
        return resolved
    for suffix in fallback_suffixes:
        resolved = _case_insensitive_file(header_path.with_suffix(suffix))
        if resolved is not None:
            return resolved
    raise FileNotFoundError(f"BrainVision referenced file not found: {candidate}")


def _case_insensitive_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    parent = path.parent
    if not parent.is_dir():
        return None
    expected = path.name.casefold()
    return next((candidate for candidate in parent.iterdir() if candidate.name.casefold() == expected), None)


def _build_eeg(
    header_path: Path,
    *,
    data_file: str,
    data: np.ndarray,
    metadata: bool,
    chanlocs: list[dict[str, Any]],
    events: list[dict[str, Any]],
    point_count: int,
    srate: float,
    common: dict[str, str],
    command: str,
    selected_channels: list[int],
    data_format: str,
    orientation: str,
) -> dict[str, Any]:
    data, events, point_count, trials, xmin = _segment_data(
        data,
        events,
        point_count=point_count,
        srate=srate,
        segmentation_type=common.get("segmentationtype", ""),
    )
    eeg = eeg_emptyset()
    eeg.update(
        {
            "setname": header_path.stem,
            "filename": header_path.name,
            "filepath": str(header_path.parent),
            "comments": f"Original file: {data_file}",
            "nbchan": len(chanlocs),
            "trials": trials,
            "pnts": point_count,
            "srate": srate,
            "xmin": xmin,
            "xmax": xmin + (point_count - 1) / srate,
            "times": (xmin + np.arange(point_count, dtype=float) / srate) * 1000.0,
            "data": data,
            "chanlocs": np.asarray(chanlocs, dtype=object),
            "urchanlocs": np.array([], dtype=object),
            "event": np.asarray(events, dtype=object),
            "urevent": np.asarray(
                [{key: value for key, value in event.items() if key != "urevent"} for event in events],
                dtype=object,
            ),
            "eventdescription": np.array([], dtype=object),
            "epoch": np.array([], dtype=object),
            "epochdescription": np.array([], dtype=object),
            "icachansind": np.array([], dtype=int),
            "session": "",
            "run": "",
            "specdata": {},
            "specicaact": {},
            "ref": "common",
            "history": command,
            "saved": "no",
            "etc": {
                "brainvision": {
                    "data_format": data_format,
                    "data_orientation": orientation,
                    "metadata_only": bool(metadata),
                    "segmentation_type": common.get("segmentationtype", ""),
                    "source_channels": np.asarray(selected_channels, dtype=int),
                }
            },
        }
    )
    checked = eeg_checkset(eeg)
    if metadata:
        checked["data"] = np.array([], dtype=float)
    return checked


def _segment_data(
    data: np.ndarray,
    events: list[dict[str, Any]],
    *,
    point_count: int,
    srate: float,
    segmentation_type: str,
) -> tuple[np.ndarray, list[dict[str, Any]], int, int, float]:
    if segmentation_type.strip().casefold() not in {"markerbased", "fixtime"}:
        return data, events, point_count, 1, 0.0
    boundary_latencies = sorted({float(event["latency"]) for event in events if event.get("type") == "boundary"})
    if len(boundary_latencies) < 2 or boundary_latencies[0] != 1:
        return data, events, point_count, 1, 0.0
    intervals = np.diff([*boundary_latencies, point_count + 1])
    if not np.allclose(intervals, intervals[0], rtol=0, atol=1e-9):
        return data, events, point_count, 1, 0.0
    epoch_points = int(round(float(intervals[0])))
    trials = len(boundary_latencies)
    if epoch_points < 1 or epoch_points * trials != point_count:
        return data, events, point_count, 1, 0.0

    kept_events = [dict(event) for event in events if event.get("type") != "boundary"]
    xmin = 0.0
    time_zero_events = [event for event in kept_events if str(event.get("code", "")).casefold() == "time 0"]
    if time_zero_events:
        for event in time_zero_events:
            event["type"] = "TLE"
        xmin = -(float(time_zero_events[0]["latency"]) - 1.0) / srate
    for index, event in enumerate(kept_events):
        event["epoch"] = int(np.ceil(float(event["latency"]) / epoch_points))
        event["urevent"] = index
    if data.size:
        data = data.reshape(data.shape[0], trials, epoch_points).transpose(0, 2, 1)
    return data, kept_events, epoch_points, trials, xmin


def _history_command(header_path: Path, srange: Any, chans: Any, metadata: bool) -> str:
    arguments = [
        format_history_value(header_path.parent),
        format_history_value(header_path.name),
    ]
    if srange is not None or chans is not None or metadata:
        arguments.append(_history_selection(srange, "srange"))
    if chans is not None or metadata:
        arguments.append(_history_selection(chans, "chans"))
    if metadata:
        arguments.append("true")
    return f"EEG = pop_loadbv({', '.join(arguments)});"


def _history_selection(value: Any, name: str) -> str:
    values = _integer_vector(value, name)
    if value is not None and np.asarray(value).ndim == 0 and values:
        return format_history_value(values[0])
    return format_history_value(values)


__all__ = ["pop_loadbv"]

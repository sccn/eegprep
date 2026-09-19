"""Read Neuroscan continuous CNT recordings."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, BinaryIO
import warnings

import numpy as np

from eegprep.functions.adminfunc.storage import MemmapData
from eegprep.functions.miscfunc._validation import integer_scalar
from eegprep.functions.miscfunc.value_parsing import parse_key_value_args


_SETUP_SIZE = 900
_CHANNEL_SIZE = 75
_NCHANNELS_OFFSET = 370
_RATE_OFFSET = 376
_NSAMPLES_OFFSET = 864
_EVENT_TABLE_OFFSET = 886
_CONTINUOUS_SECONDS_OFFSET = 890
_CHANNEL_OFFSET = 894
_EVENT_HEADER = struct.Struct("<BII")
_EVENT_FORMATS = {
    1: struct.Struct("<HBBi"),
    2: struct.Struct("<HBBihhfbbb"),
    3: struct.Struct("<HBBihhfbbb"),
}
_LOAD_OPTIONS = {
    "t1",
    "sample1",
    "lddur",
    "ldnsamples",
    "scale",
    "dataformat",
    "data_format",
    "blockread",
    "memmapfile",
    "precision",
    "keystroke",
}


def loadcnt(
    filename: str | Path,
    *args: Any,
    t1: Any = 0,
    sample1: Any = None,
    lddur: Any = None,
    ldnsamples: Any = None,
    scale: Any = "on",
    dataformat: Any = "auto",
    blockread: Any = None,
    memmapfile: str | Path | None = None,
    precision: Any = "single",
    **kwargs: Any,
) -> dict[str, Any]:
    """Load a Neuroscan CNT file into a low-level CNT dictionary.

    Parameters use the names exposed by EEGLAB's ``loadcnt``. ``sample1`` is
    a zero-based starting sample and overrides ``t1``; ``ldnsamples`` overrides
    ``lddur``. Data are channel-major and scaled to microvolts by default.

    Args:
        filename: Neuroscan CNT file.
        *args: Optional EEGLAB-style key/value pairs.
        t1: Start time in seconds.
        sample1: Zero-based starting sample.
        lddur: Duration to read in seconds.
        ldnsamples: Number of samples to read.
        scale: ``"on"`` for microvolts or ``"off"`` for stored counts.
        dataformat: ``"auto"``, ``"int16"``, or ``"int32"``.
        blockread: Optional byte block size override; ``1`` means interleaved.
        memmapfile: Optional ``.fdt`` path for float32 disk-backed output.
        precision: In-memory output precision, ``"single"`` or ``"double"``.
        **kwargs: ``data_format`` alias and legacy ``keystroke`` option.

    Returns:
        Dictionary containing ``header``, ``electloc``, ``data``, ``event``,
        ``dataformat``, ``ldnsamples``, ``labels``, ``Teeg``, and ``tag``.
    """
    options = {
        "t1": t1,
        "sample1": sample1,
        "lddur": lddur,
        "ldnsamples": ldnsamples,
        "scale": scale,
        "dataformat": dataformat,
        "blockread": blockread,
        "memmapfile": memmapfile,
        "precision": precision,
    }
    options.update(parse_key_value_args(args, kwargs, lowercase_kwargs=True))
    unknown = set(options) - _LOAD_OPTIONS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unsupported loadcnt option(s): {names}")
    if "data_format" in options:
        if options.get("dataformat", "auto") != "auto":
            raise ValueError("Use only one of dataformat and data_format")
        options["dataformat"] = options.pop("data_format")

    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CNT file not found: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        setup = _read_exact(stream, _SETUP_SIZE, "CNT setup header")
        header = _parse_header(setup)
        _validate_header(header)
        channel_bytes = _read_exact(
            stream,
            int(header["nchannels"]) * _CHANNEL_SIZE,
            "CNT channel headers",
        )
        electloc = _parse_electrodes(channel_bytes, int(header["nchannels"]))

    data_offset = _SETUP_SIZE + _CHANNEL_SIZE * int(header["nchannels"])
    event_position = _resolve_event_position(header, file_size, data_offset)
    resolved_format = _resolve_data_format(path, header, data_offset, event_position, options["dataformat"])
    bytes_per_sample = 2 if resolved_format == "int16" else 4
    total_samples = _sample_count(
        header,
        data_offset=data_offset,
        event_position=event_position,
        bytes_per_sample=bytes_per_sample,
    )
    first_sample, sample_count = _selection(
        rate=float(header["rate"]),
        total_samples=total_samples,
        t1=options["t1"],
        sample1=options["sample1"],
        lddur=options["lddur"],
        ldnsamples=options["ldnsamples"],
    )
    output_precision = _precision(options["precision"])
    scale_data = _toggle(options["scale"], "scale")
    block_samples = _block_samples(header["channeloffset"], options["blockread"], bytes_per_sample)
    data = _read_data(
        path,
        data_offset=data_offset,
        channels=int(header["nchannels"]),
        total_samples=total_samples,
        first_sample=first_sample,
        sample_count=sample_count,
        dataformat=resolved_format,
        block_samples=block_samples,
        electrodes=electloc,
        scale=scale_data,
        precision=output_precision,
        memmapfile=options["memmapfile"],
    )
    events, event_header = _read_events(
        path,
        event_position=event_position,
        file_size=file_size,
        data_offset=data_offset,
        channels=int(header["nchannels"]),
        bytes_per_sample=bytes_per_sample,
        first_sample=first_sample,
        sample_count=sample_count,
    )
    header["dataformat"] = resolved_format
    header["samples"] = total_samples
    header["block_samples"] = block_samples
    return {
        "header": header,
        "electloc": electloc,
        "data": data,
        "event": events,
        "dataformat": resolved_format,
        "ldnsamples": sample_count,
        "sample1": first_sample,
        "labels": [electrode["lab"] for electrode in electloc],
        "Teeg": event_header,
        "tag": _last_byte(path),
    }


def _parse_header(setup: bytes) -> dict[str, Any]:
    return {
        "rev": _text(setup[0:12]),
        "nextfile": _unpack("<i", setup, 12),
        "prevfile": _unpack("<I", setup, 16),
        "type": _unpack("<b", setup, 20),
        "id": _text(setup[21:41]),
        "oper": _text(setup[41:61]),
        "doctor": _text(setup[61:81]),
        "hospital": _text(setup[101:121]),
        "patient": _text(setup[121:141]),
        "age": _unpack("<h", setup, 141),
        "sex": _text(setup[143:144]),
        "hand": _text(setup[144:145]),
        "label": _text(setup[205:225]),
        "date": _text(setup[225:235]),
        "time": _text(setup[235:247]),
        "nchannels": _unpack("<H", setup, _NCHANNELS_OFFSET),
        "rate": _unpack("<H", setup, _RATE_OFFSET),
        "numsamples": _unpack("<I", setup, _NSAMPLES_OFFSET),
        "eventtablepos": _unpack("<I", setup, _EVENT_TABLE_OFFSET),
        "continuousseconds": _unpack("<f", setup, _CONTINUOUS_SECONDS_OFFSET),
        "channeloffset": _unpack("<i", setup, _CHANNEL_OFFSET),
    }


def _validate_header(header: dict[str, Any]) -> None:
    if int(header["nchannels"]) <= 0:
        raise ValueError("CNT header reports no channels")
    if float(header["rate"]) <= 0:
        raise ValueError("CNT header reports a non-positive sampling rate")
    if int(header["channeloffset"]) < 0:
        raise ValueError("CNT header reports a negative channel block offset")


def _parse_electrodes(data: bytes, channels: int) -> list[dict[str, Any]]:
    electrodes = []
    for index in range(channels):
        record = data[index * _CHANNEL_SIZE : (index + 1) * _CHANNEL_SIZE]
        electrodes.append(
            {
                "lab": _text(record[0:10]),
                "reference": _unpack("<b", record, 10),
                "skip": _unpack("<b", record, 11),
                "reject": _unpack("<b", record, 12),
                "display": _unpack("<b", record, 13),
                "bad": _unpack("<b", record, 14),
                "n": _unpack("<H", record, 15),
                "avg_reference": _unpack("<b", record, 17),
                "x_coord": _unpack("<f", record, 19),
                "y_coord": _unpack("<f", record, 23),
                "baseline": _unpack("<h", record, 47),
                "senstivity": _unpack("<f", record, 59),
                "gain": _unpack("<b", record, 63),
                "calib": _unpack("<f", record, 71),
            }
        )
    return electrodes


def _resolve_event_position(header: dict[str, Any], file_size: int, data_offset: int) -> int:
    low = int(header["eventtablepos"])
    high = int(header["prevfile"])
    combined = low + high * 2**32
    candidates = [combined, low] if high else [low]
    for candidate in candidates:
        if data_offset <= candidate <= file_size:
            return candidate
    if low == 0 and int(header["numsamples"]) > 0:
        return file_size
    raise ValueError("CNT event-table position is outside the file")


def _resolve_data_format(
    path: Path,
    header: dict[str, Any],
    data_offset: int,
    event_position: int,
    requested: Any,
) -> str:
    value = str(requested).strip().lower()
    if value in {"int16", "int32"}:
        return value
    if value != "auto":
        raise ValueError("dataformat must be 'auto', 'int16', or 'int32'")

    channels = int(header["nchannels"])
    header_samples = int(header["numsamples"])
    data_bytes = event_position - data_offset
    if header_samples > 0:
        bytes_per_sample, remainder = divmod(data_bytes, channels * header_samples)
        if remainder == 0 and bytes_per_sample in {2, 4}:
            return f"int{bytes_per_sample * 8}"

    nextfile = int(header["nextfile"])
    if nextfile > 0 and nextfile + 52 < path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(nextfile + 52)
            flag = stream.read(1)
        if flag == b"\x01":
            return "int32"
        if flag == b"\x00":
            return "int16"

    if data_bytes % (channels * 2) == 0:
        warnings.warn(
            "CNT sample width is ambiguous; using the Neuroscan/EEGLAB int16 fallback. "
            "Pass dataformat explicitly to prevent ambiguity.",
            RuntimeWarning,
            stacklevel=3,
        )
        return "int16"
    raise ValueError("CNT data size cannot be interpreted as int16 or int32 samples")


def _sample_count(
    header: dict[str, Any],
    *,
    data_offset: int,
    event_position: int,
    bytes_per_sample: int,
) -> int:
    frame_bytes = int(header["nchannels"]) * bytes_per_sample
    samples, remainder = divmod(event_position - data_offset, frame_bytes)
    if remainder:
        raise ValueError("CNT data section is not an exact number of channel frames")
    if samples <= 0:
        raise ValueError("CNT file contains no data samples")
    header_samples = int(header["numsamples"])
    if header_samples > 0 and header_samples != samples:
        warnings.warn(
            f"CNT header reports {header_samples} samples but the data section contains {samples}; "
            "using the data-section length.",
            RuntimeWarning,
            stacklevel=3,
        )
    return samples


def _selection(
    *,
    rate: float,
    total_samples: int,
    t1: Any,
    sample1: Any,
    lddur: Any,
    ldnsamples: Any,
) -> tuple[int, int]:
    if sample1 is None or sample1 == "":
        start_seconds = _finite_number(t1, "t1")
        if start_seconds < 0:
            raise ValueError("t1 must be non-negative")
        start = _nearest_sample(start_seconds * rate)
    else:
        start = integer_scalar(sample1, "sample1")
        if start < 0:
            raise ValueError("sample1 must be non-negative")
    if start >= total_samples:
        raise ValueError("Requested CNT start sample is outside the recording")

    if ldnsamples is not None and ldnsamples != "":
        count = integer_scalar(ldnsamples, "ldnsamples")
    elif lddur is not None and lddur != "":
        duration = _finite_number(lddur, "lddur")
        count = _nearest_sample(duration * rate)
    else:
        count = total_samples - start
    if count <= 0:
        raise ValueError("Requested CNT sample count must be positive")
    available = total_samples - start
    if count > available:
        warnings.warn(
            f"Requested {count} CNT samples from sample {start}, but only {available} remain; truncating.",
            RuntimeWarning,
            stacklevel=3,
        )
        count = available
    return start, count


def _block_samples(header_value: Any, override: Any, bytes_per_sample: int) -> int:
    value = int(header_value) if override is None or override == "" else integer_scalar(override, "blockread")
    if value <= 1:
        return 1
    blocks, remainder = divmod(value, bytes_per_sample)
    if remainder or blocks <= 0:
        raise ValueError("CNT channel block offset must be divisible by the sample width")
    return blocks


def _read_data(
    path: Path,
    *,
    data_offset: int,
    channels: int,
    total_samples: int,
    first_sample: int,
    sample_count: int,
    dataformat: str,
    block_samples: int,
    electrodes: list[dict[str, Any]],
    scale: bool,
    precision: np.dtype[Any],
    memmapfile: Any,
) -> np.ndarray | MemmapData:
    mapped_path = None if memmapfile is None or str(memmapfile) == "" else Path(memmapfile).expanduser()
    if mapped_path is not None and mapped_path.suffix.lower() != ".fdt":
        raise ValueError("memmapfile must use the .fdt suffix")
    target_dtype = np.dtype("<f4") if mapped_path is not None else precision
    if mapped_path is None:
        target: np.ndarray = np.empty((channels, sample_count), dtype=target_dtype, order="F")
    else:
        mapped_path.parent.mkdir(parents=True, exist_ok=True)
        target = np.memmap(mapped_path, dtype=target_dtype, mode="w+", shape=(channels, sample_count), order="F")

    source_dtype = np.dtype("<i2" if dataformat == "int16" else "<i4")
    baselines = np.asarray([item["baseline"] for item in electrodes], dtype=target_dtype)
    factors = np.asarray(
        [item["senstivity"] * item["calib"] / 204.8 for item in electrodes],
        dtype=target_dtype,
    )

    def store(values: np.ndarray, source_start: int) -> None:
        lower = max(first_sample, source_start)
        upper = min(first_sample + sample_count, source_start + values.shape[1])
        if lower >= upper:
            return
        selected = values[:, lower - source_start : upper - source_start].astype(target_dtype)
        if scale:
            selected = (selected - baselines[:, np.newaxis]) * factors[:, np.newaxis]
        target[:, lower - first_sample : upper - first_sample] = selected

    full_blocks, tail_samples = divmod(total_samples, block_samples)
    bytes_per_block = channels * block_samples * source_dtype.itemsize
    first_block = first_sample // block_samples
    last_block = min(full_blocks, (first_sample + sample_count + block_samples - 1) // block_samples)
    blocks_per_chunk = max(1, 64 * 1024**2 // max(bytes_per_block, 1))
    with path.open("rb") as stream:
        block = first_block
        while block < last_block:
            chunk_blocks = min(blocks_per_chunk, last_block - block)
            stream.seek(data_offset + block * bytes_per_block)
            payload = _read_exact(stream, chunk_blocks * bytes_per_block, "CNT sample block")
            values = np.frombuffer(payload, dtype=source_dtype)
            values = values.reshape(chunk_blocks, channels, block_samples).transpose(1, 0, 2)
            store(values.reshape(channels, chunk_blocks * block_samples), block * block_samples)
            block += chunk_blocks
        if tail_samples and first_sample + sample_count > full_blocks * block_samples:
            stream.seek(data_offset + full_blocks * bytes_per_block)
            payload = _read_exact(
                stream,
                channels * tail_samples * source_dtype.itemsize,
                "CNT final sample block",
            )
            store(
                np.frombuffer(payload, dtype=source_dtype).reshape(channels, tail_samples), full_blocks * block_samples
            )

    if mapped_path is None:
        return target
    if isinstance(target, np.memmap):
        target.flush()
    return MemmapData(mapped_path, (channels, sample_count), dtype="<f4", mode="r+", order="F")


def _read_events(
    path: Path,
    *,
    event_position: int,
    file_size: int,
    data_offset: int,
    channels: int,
    bytes_per_sample: int,
    first_sample: int,
    sample_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if event_position == file_size:
        return [], {"teeg": 0, "size": 0, "offset": 0}
    with path.open("rb") as stream:
        stream.seek(event_position)
        header_data = _read_exact(stream, _EVENT_HEADER.size, "CNT event-table header")
        event_type, size, offset = _EVENT_HEADER.unpack(header_data)
        parser = _EVENT_FORMATS.get(event_type)
        if parser is None:
            raise ValueError(f"Unsupported CNT event-table type: {event_type}")
        if size % parser.size:
            raise ValueError("CNT event-table size is not a whole number of records")
        if event_position + _EVENT_HEADER.size + size > file_size:
            raise ValueError("CNT event table is truncated")
        payload = _read_exact(stream, size, "CNT event table")

    events = []
    for fields in parser.iter_unpack(payload):
        stimulus, keyboard, keypad_byte, stored_offset = fields[:4]
        if event_type == 3:
            # Type 3 stores a zero-based global sample frame rather than a
            # byte offset. Normalize every table type to a 1-based sample.
            sample_number = float(stored_offset) + 1.0
        else:
            sample_number = (float(stored_offset) - data_offset) / (channels * bytes_per_sample)
        if sample_number != int(sample_number):
            raise ValueError("CNT event offset is not aligned to a complete channel frame")
        selected_sample = sample_number - first_sample
        if selected_sample < 1 or selected_sample > sample_count:
            continue
        event = {
            "stimtype": int(stimulus),
            "keyboard": int(keyboard),
            "keypad_accept": int(keypad_byte) & 0x0F,
            "accept_ev1": int(keypad_byte) >> 4,
            "offset": selected_sample,
        }
        if event_type in {2, 3}:
            event.update(
                {
                    "type": int(fields[4]),
                    "code": int(fields[5]),
                    "latency": float(fields[6]),
                    "epochevent": int(fields[7]),
                    "accept": int(fields[8]),
                    "accuracy": int(fields[9]),
                }
            )
        events.append(event)
    return events, {"teeg": int(event_type), "size": int(size), "offset": int(offset)}


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"{label} is truncated: expected {size} bytes, found {len(data)}")
    return data


def _unpack(format_string: str, data: bytes, offset: int) -> Any:
    return struct.unpack_from(format_string, data, offset)[0]


def _text(data: bytes) -> str:
    return data.split(b"\x00", 1)[0].decode("latin-1").strip()


def _last_byte(path: Path) -> int:
    with path.open("rb") as stream:
        stream.seek(-1, 2)
        return int(stream.read(1)[0])


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nearest_sample(value: float) -> int:
    return int(np.floor(value + 0.5))


def _precision(value: Any) -> np.dtype[Any]:
    name = str(value).strip().lower()
    if name == "single":
        return np.dtype(np.float32)
    if name == "double":
        return np.dtype(np.float64)
    raise ValueError("precision must be 'single' or 'double'")


def _toggle(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "on":
        return True
    if text == "off":
        return False
    raise ValueError(f"{name} must be 'on' or 'off'")


__all__ = ["loadcnt"]

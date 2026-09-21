"""Read EGI Simple Binary EEG samples and event channels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from eegprep.functions.sigprocfunc.readegihdr import _read_egi_header


def readegi(
    filename: str | Path,
    data_chunks: int | Sequence[int] | np.ndarray | None = None,
    forceversion: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Read an EGI Simple Binary file (versions 2 through 7).

    Args:
        filename: EGI ``.RAW`` file.
        data_chunks: Optional 1-based frame numbers for continuous data or
            segment numbers for segmented data. Selected chunks are returned
            in recording order, as in EEGLAB. Empty input selects the whole
            recording.
        forceversion: Optional header-version override from 2 through 7.

    Returns:
        ``(header, trial_data, event_data, segment_category_indices)``.
        Low-level data arrays are channel-major with selected segments
        concatenated along the sample axis, matching EEGLAB's ``readegi``.
    """
    path = Path(filename)
    with path.open("rb") as stream:
        header = _read_egi_header(stream, forceversion)
        total_chunks = header["segments"] if header["segmented"] else header["samples"]
        selected = _chunk_indices(data_chunks, total_chunks)
        if header["segmented"]:
            values, categories, start_times = _read_segmented(stream, header, selected)
            header["segment_start_times"] = start_times
        else:
            values = _read_continuous(stream, header, selected)
            categories = np.array([], dtype=np.int64)
            header["segment_start_times"] = np.array([], dtype=np.int64)

    nchan = int(header["nchan"])
    trial_data = values[:nchan].astype(np.float64, copy=False)
    event_data = values[nchan:].astype(np.float64, copy=False)
    if header["bits"] != 0 and header["range"] != 0:
        trial_data = trial_data * (float(header["range"]) / (2.0 ** int(header["bits"])))
    return header, trial_data, event_data, categories


def _read_continuous(stream: Any, header: dict[str, Any], selected: np.ndarray) -> np.ndarray:
    frame_values = int(header["nchan"] + header["eventtypes"])
    frame_bytes = frame_values * int(header["sample_width"])
    if selected.size == header["samples"] and np.array_equal(selected, np.arange(header["samples"])):
        raw = _read_exact(stream, frame_bytes * selected.size, "continuous EGI samples")
        return _decode_samples(raw, header, frame_values, selected.size)

    output = np.empty((frame_values, selected.size), dtype=np.float64)
    for output_index, frame_index in enumerate(selected):
        stream.seek(int(header["header_bytes"]) + int(frame_index) * frame_bytes)
        raw = _read_exact(stream, frame_bytes, f"EGI frame {int(frame_index) + 1}")
        output[:, output_index] = np.frombuffer(raw, dtype=header["sample_dtype"])
    return output


def _read_segmented(
    stream: Any,
    header: dict[str, Any],
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_values = int(header["nchan"] + header["eventtypes"])
    samples_per_segment = int(header["segsamps"])
    payload_bytes = frame_values * samples_per_segment * int(header["sample_width"])
    record_bytes = 6 + payload_bytes
    blocks = []
    categories = np.empty(selected.size, dtype=np.int64)
    start_times = np.empty(selected.size, dtype=np.int64)
    for output_index, segment_index in enumerate(selected):
        stream.seek(int(header["header_bytes"]) + int(segment_index) * record_bytes)
        prefix = _read_exact(stream, 6, f"EGI segment {int(segment_index) + 1} prefix")
        categories[output_index] = int.from_bytes(prefix[:2], "big", signed=True)
        start_times[output_index] = int.from_bytes(prefix[2:], "big", signed=True)
        raw = _read_exact(stream, payload_bytes, f"EGI segment {int(segment_index) + 1} samples")
        blocks.append(_decode_samples(raw, header, frame_values, samples_per_segment))
    empty = np.empty((frame_values, 0), dtype=np.float64)
    return (np.concatenate(blocks, axis=1) if blocks else empty), categories, start_times


def _decode_samples(
    raw: bytes,
    header: dict[str, Any],
    frame_values: int,
    samples: int,
) -> np.ndarray:
    values = np.frombuffer(raw, dtype=header["sample_dtype"])
    expected = frame_values * samples
    if values.size != expected:
        raise ValueError(f"Expected {expected} EGI values but decoded {values.size}")
    return values.reshape(samples, frame_values).T.astype(np.float64, copy=False)


def _chunk_indices(data_chunks: int | Sequence[int] | np.ndarray | None, count: int) -> np.ndarray:
    if data_chunks is None:
        return np.arange(count, dtype=np.int64)
    chunks = np.asarray(data_chunks)
    if chunks.size == 0:
        return np.arange(count, dtype=np.int64)
    if chunks.ndim > 2 or (chunks.ndim == 2 and 1 not in chunks.shape):
        raise ValueError("data_chunks must be empty or a vector")
    flat = chunks.reshape(-1)
    if flat.dtype.kind not in "iuf" or np.issubdtype(flat.dtype, np.bool_):
        raise ValueError("data_chunks must contain integer 1-based indices")
    try:
        numeric = flat.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("data_chunks must contain integer 1-based indices") from error
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("data_chunks must contain integer 1-based indices")
    indices = numeric.astype(np.int64) - 1
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError(f"data_chunks must be within the 1-based range 1..{count}")
    if np.unique(indices).size != indices.size:
        raise ValueError("data_chunks must not contain duplicate indices")
    return np.sort(indices)


def _read_exact(stream: Any, size: int, field: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of file while reading {field}")
    return data


__all__ = ["readegi"]

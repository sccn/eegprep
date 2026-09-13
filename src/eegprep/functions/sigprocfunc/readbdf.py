"""Read selected records from BioSemi Data Format files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.miscfunc._validation import integer_array, integer_scalar


_BDF_BYTES_PER_SAMPLE = 3


def readbdf(
    dataset: dict[str, Any],
    records: Any,
    mode: int = 0,
) -> tuple[dict[str, Any], np.ndarray]:
    """Read selected 1-based BDF records.

    Modes 0 and 2 apply physical calibration; odd modes retain raw digital
    values. Modes 2 and 3 compact each channel's variable-length records at
    that channel's own samples-per-record stride, matching EEGLAB.

    Args:
        dataset: Structure returned by :func:`openbdf`.
        records: Scalar or array of EEGLAB-facing 1-based record numbers.
        mode: Legacy record layout and calibration mode (0 through 3).

    Returns:
        ``(updated_dataset, last_raw_record)``. ``Record`` in the updated
        dataset is channel-major; the raw record is sample-major.
    """
    if not isinstance(dataset, dict) or not isinstance(dataset.get("Head"), dict):
        raise TypeError("dataset must be a structure returned by openbdf")
    read_mode = integer_scalar(mode, "mode")
    if read_mode not in {0, 1, 2, 3}:
        raise ValueError("mode must be 0, 1, 2, or 3")
    indices = integer_array(records, "records").reshape(-1)
    head = dataset["Head"]
    record_count = int(head["NRec"])
    if np.any(indices < 1) or np.any(indices > record_count):
        raise IndexError(f"record numbers must be between 1 and {record_count}")

    samples_per_record = np.asarray(head["SPR"], dtype=np.intp)
    signal_count = int(head["NS"])
    if samples_per_record.shape != (signal_count,):
        raise ValueError("BDF header SPR does not match NS")
    maximum_samples = int(samples_per_record.max())
    samples_in_block = int(samples_per_record.sum())
    record_bytes = samples_in_block * _BDF_BYTES_PER_SAMPLE
    output = np.zeros((indices.size * maximum_samples, signal_count), dtype=float)
    valid = np.zeros(indices.size * maximum_samples, dtype=np.uint8)
    last_raw = np.full((maximum_samples, signal_count), np.nan, dtype=float)
    warnings = np.zeros(signal_count, dtype=np.uint8)

    path = _data_path(head)
    with path.open("rb") as stream:
        for output_record, one_based_record in enumerate(indices):
            stream.seek(int(head["HeadLen"]) + (int(one_based_record) - 1) * record_bytes)
            packed = stream.read(record_bytes)
            if len(packed) != record_bytes:
                raise ValueError(f"BDF file is incomplete while reading record {one_based_record}")
            decoded = _decode_signed_24(packed)
            last_raw = _record_matrix(decoded, samples_per_record, maximum_samples)
            _store_record(output, last_raw, samples_per_record, output_record, maximum_samples, compact=read_mode >= 2)

            selected = np.asarray(head["Chan_Select"], dtype=bool)
            selected_values = last_raw[:, selected]
            selected_minimum = np.asarray(head["DigMin"], dtype=float)[selected]
            selected_maximum = np.asarray(head["DigMax"], dtype=float)[selected]
            record_valid = np.all(
                (selected_values > selected_minimum[None, :]) & (selected_values < selected_maximum[None, :]),
                axis=1,
            )
            start = output_record * maximum_samples
            valid[start : start + maximum_samples] = record_valid.astype(np.uint8)
            for channel, count in enumerate(samples_per_record):
                values = last_raw[:count, channel]
                warnings[channel] |= np.uint8(
                    np.any(values > head["DigMax"][channel]) or np.any(values < head["DigMin"][channel])
                )

    if read_mode % 2 == 0:
        output = output * np.asarray(head["Cal"], dtype=float)[None, :] + np.asarray(head["Off"], dtype=float)[None, :]

    result = deepcopy(dataset)
    result["Record"] = output.T
    result["Valid"] = valid[None, :]
    result["Idx"] = indices.copy()
    result["Head"].setdefault("ERROR", {})["DigMinMax_Warning"] = warnings
    return result, last_raw


def _data_path(head: dict[str, Any]) -> Path:
    if "FileName" in head:
        return Path(head["FileName"])
    file_info = head["FILE"]
    suffix = f".{file_info['Ext']}" if file_info.get("Ext") else ""
    return Path(file_info["Path"]) / f"{file_info['Name']}{suffix}"


def _decode_signed_24(packed: bytes) -> np.ndarray:
    octets = np.frombuffer(packed, dtype=np.uint8).reshape(-1, _BDF_BYTES_PER_SAMPLE).astype(np.int32)
    values = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
    values[values >= 1 << 23] -= 1 << 24
    return values


def _record_matrix(decoded: np.ndarray, samples_per_record: np.ndarray, maximum_samples: int) -> np.ndarray:
    record = np.full((maximum_samples, samples_per_record.size), np.nan, dtype=float)
    position = 0
    for channel, count in enumerate(samples_per_record):
        record[:count, channel] = decoded[position : position + count]
        position += count
    return record


def _store_record(
    output: np.ndarray,
    record: np.ndarray,
    samples_per_record: np.ndarray,
    output_record: int,
    maximum_samples: int,
    *,
    compact: bool,
) -> None:
    if not compact:
        start = output_record * maximum_samples
        output[start : start + maximum_samples] = record
        return
    for channel, count in enumerate(samples_per_record):
        start = output_record * count
        output[start : start + count, channel] = record[:count, channel]


__all__ = ["readbdf"]

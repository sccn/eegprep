"""Compare EEGPrep binary readers with EEGLAB results from real recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def require_matlab_reference(request) -> None:
    """Skip unless this process is comparing against MATLAB EEGLAB."""
    if request.config.getoption("--eeglab-backend") != "matlab":
        pytest.skip("real-file EEGLAB comparison requires --eeglab-backend=matlab")


def require_real_recording(path: Path) -> Path:
    """Return a suite recording, failing when the checkout has only an LFS pointer."""
    if not path.is_file():
        pytest.fail(f"Missing EEGLAB test recording: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(len(_LFS_POINTER))
    if prefix.startswith(_LFS_POINTER):
        pytest.fail(f"{path} is a Git LFS pointer, not the recording")
    return path


def assert_same_samples(actual, expected, *, label: str) -> None:
    """Require identical numeric samples, including NaNs, and report the max gap."""
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(f"{label} shape {actual_array.shape} != {expected_array.shape}")
    if actual_array.size == 0 or np.array_equal(actual_array, expected_array, equal_nan=True):
        return
    difference = np.nanmax(np.abs(actual_array - expected_array))
    raise AssertionError(f"{label} values differ; max abs diff {difference}")


def assert_same_eeg(actual: dict, expected: dict, *, label: str) -> None:
    """Compare signal, dimensions, channel labels, and event type/latency."""
    assert_same_samples(actual["data"], expected["data"], label=f"{label} data")
    for field in ("srate", "nbchan", "pnts", "trials", "xmin", "xmax"):
        actual_value = _number(actual[field])
        expected_value = _number(expected[field])
        if actual_value != expected_value:
            raise AssertionError(f"{label} {field} {actual_value} != {expected_value}")
    actual_labels = _labels(actual.get("chanlocs"))
    expected_labels = _labels(expected.get("chanlocs"))
    if actual_labels != expected_labels:
        raise AssertionError(f"{label} channels {actual_labels} != {expected_labels}")
    actual_events = _events(actual.get("event"))
    expected_events = _events(expected.get("event"))
    if actual_events != expected_events:
        raise AssertionError(f"{label} events {actual_events} != {expected_events}")


def assert_same_cnt(actual: dict, expected: dict, *, label: str) -> None:
    """Compare low-level CNT samples, labels, and stimulus offsets."""
    assert_same_samples(actual["data"], expected["data"], label=f"{label} data")
    actual_header = _mapping(actual["header"])
    expected_header = _mapping(expected["header"])
    if _number(actual_header["rate"]) != _number(expected_header["rate"]):
        raise AssertionError(f"{label} rate differs")
    if _number(actual["ldnsamples"]) != _number(expected["ldnsamples"]):
        raise AssertionError(f"{label} ldnsamples differs")
    actual_labels = _string_list(actual["labels"])
    expected_labels = _string_list(expected["labels"])
    if actual_labels != expected_labels:
        raise AssertionError(f"{label} channels {actual_labels} != {expected_labels}")
    actual_events = _cnt_events(actual.get("event"))
    expected_events = _cnt_events(expected.get("event"))
    if actual_events != expected_events:
        raise AssertionError(f"{label} events {actual_events} != {expected_events}")


def assert_same_bdf_header(actual: dict, expected: dict, *, label: str) -> None:
    """Compare BDF header fields that describe the recording."""
    for field in ("NRec", "NS", "Dur"):
        if _number(actual[field]) != _number(expected[field]):
            raise AssertionError(f"{label} {field} differs")
    for field in ("SPR", "Cal", "Off", "DigMin", "DigMax", "PhysMin", "PhysMax", "SampleRate"):
        assert_same_samples(
            np.ravel(np.asarray(actual[field])),
            np.ravel(np.asarray(expected[field])),
            label=f"{label} {field}",
        )
    actual_labels = _string_list(actual["Label"])
    expected_labels = _string_list(expected["Label"])
    if actual_labels != expected_labels:
        raise AssertionError(f"{label} labels {actual_labels} != {expected_labels}")
    if _token(actual["ChanTyp"]) != _token(expected["ChanTyp"]):
        raise AssertionError(f"{label} ChanTyp {_token(actual['ChanTyp'])!r} != {_token(expected['ChanTyp'])!r}")


def _records(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, np.ndarray) and value.size == 0:
        return []
    if isinstance(value, np.ndarray) and value.dtype.names:
        return [{name: record[name] for name in value.dtype.names} for record in np.ravel(value, order="F")]
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel(order="F"))
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    records = _records(value)
    if len(records) == 1 and isinstance(records[0], dict):
        return records[0]
    raise AssertionError(f"expected one structure, got {type(value).__name__}")


def _number(value) -> float | int:
    array = np.asarray(value)
    if array.dtype == object:
        if array.size != 1:
            raise AssertionError(f"expected one number, got {value!r}")
        return _number(array.reshape(-1)[0])
    if array.dtype.kind not in "iuf" or array.size != 1:
        raise AssertionError(f"expected one number, got {value!r}")
    return array.reshape(-1)[0].item()


def _token(value):
    if isinstance(value, str):
        return value.strip()
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        return _token(array.reshape(-1)[0])
    if array.dtype.kind in "US":
        if array.ndim > 1:
            return "".join(str(item) for item in array.ravel()).strip()
        if array.size == 1:
            return str(array.reshape(-1)[0]).strip()
    if array.size == 1 and array.dtype.kind in "iuf":
        number = float(array.reshape(-1)[0])
        if number.is_integer():
            return int(number)
        return number
    return str(value).strip()


def _string_list(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()]
    array = np.asarray(value)
    if array.dtype == object or isinstance(value, (list, tuple)):
        return [str(_token(item)) for item in np.asarray(value, dtype=object).ravel(order="F")]
    if array.dtype.kind in "US":
        if array.ndim <= 1:
            return [str(item).strip() for item in array.ravel()]
        return ["".join(str(character) for character in row).strip() for row in array]
    raise AssertionError(f"expected text, got {array.dtype}")


def _labels(value) -> list[str]:
    return [str(_token(record["labels"])) for record in _records(value)]


def _events(value) -> list[tuple]:
    return [(_token(record["type"]), float(_number(record["latency"]))) for record in _records(value)]


def _cnt_events(value) -> list[tuple[int, float]]:
    return [(int(_number(record["stimtype"])), float(_number(record["offset"]))) for record in _records(value)]

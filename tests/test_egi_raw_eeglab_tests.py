"""Behavioral ports of the current EEGLAB EGI Simple Binary tests."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from eegprep import pop_readegi, pop_readsegegi, readegi, readegihdr
from tests.eeglab_tests import eeglab_test


BINARY_SUITE = "unittesting_binary"
_SAMPLE_DTYPES = {2: ">i2", 3: ">i2", 4: ">f4", 5: ">f4", 6: ">f8", 7: ">f8"}


def _segmented_values() -> tuple[np.ndarray, np.ndarray]:
    signals = np.array(
        [
            [[1, 11, 21], [2, 12, 22], [3, 13, 23], [4, 14, 24]],
            [[31, 41, 51], [32, 42, 52], [33, 43, 53], [34, 44, 54]],
        ],
        dtype=float,
    )
    events = np.array([[[0, 0, 0], [0, 0, 0], [5, 6, 7], [0, 0, 0]]], dtype=float)
    return signals, events


def _write_egi_raw(
    path: Path,
    version: int,
    signals: np.ndarray,
    *,
    events: np.ndarray | None = None,
    event_codes: tuple[str, ...] = (),
    category_names: tuple[str, ...] = ("Eyes open", "Eyes closed"),
    category_indices: tuple[int, ...] | None = None,
    segment_start_times: tuple[int, ...] | None = None,
    sample_rate: int = 250,
    gain: int = 1,
    bits: int = 12,
    signal_range: int = 4096,
    header_version: int | None = None,
) -> None:
    segmented = version in {3, 5, 7}
    data = np.asarray(signals)
    expected_ndim = 3 if segmented else 2
    if data.ndim != expected_ndim:
        raise ValueError(f"version {version} data must have {expected_ndim} dimensions")
    nchan = data.shape[0]
    sample_shape = data.shape[1:]
    if events is None:
        event_data = np.empty((0, *sample_shape))
    else:
        event_data = np.asarray(events)
    if event_data.shape != (len(event_codes), *sample_shape):
        raise ValueError("event data shape does not match event codes and signal samples")

    encoded_version = header_version or version
    output = bytearray(
        struct.pack(
            ">i6hi5h",
            encoded_version,
            2025,
            2,
            3,
            4,
            5,
            6,
            7,
            sample_rate,
            nchan,
            gain,
            bits,
            signal_range,
        )
    )
    if segmented:
        segments = data.shape[2]
        categories = category_indices or tuple(1 for _ in range(segments))
        start_times = segment_start_times or tuple(index * 100 for index in range(segments))
        output.extend(struct.pack(">h", len(category_names)))
        for name in category_names:
            encoded = name.encode("latin-1")
            output.extend(struct.pack(">B", len(encoded)))
            output.extend(encoded)
        output.extend(struct.pack(">hi", segments, data.shape[1]))
    else:
        output.extend(struct.pack(">i", data.shape[1]))
        categories = ()
        start_times = ()
    output.extend(struct.pack(">h", len(event_codes)))
    for code in event_codes:
        output.extend(code.encode("latin-1")[:4].ljust(4, b" "))

    dtype = _SAMPLE_DTYPES[version]
    if segmented:
        for segment, (category, start_time) in enumerate(zip(categories, start_times)):
            output.extend(struct.pack(">hi", category, start_time))
            block = np.vstack((data[:, :, segment], event_data[:, :, segment]))
            output.extend(block.T.astype(dtype).tobytes())
    else:
        block = np.vstack((data, event_data))
        output.extend(block.T.astype(dtype).tobytes())
    path.write_bytes(output)


@eeglab_test(f"{BINARY_SUITE}/readegi/binary_readegi_wrapperTest.m", "test_test_readegi")
def test_readegi_decodes_selected_segments_samples_events_and_categories(tmp_path: Path) -> None:
    signals, events = _segmented_values()
    filename = tmp_path / "TESTEGI.RAW"
    _write_egi_raw(
        filename,
        3,
        signals,
        events=events,
        event_codes=("stim",),
        category_indices=(1, 2, 2),
        segment_start_times=(100, 200, 300),
    )

    header, trial_data, event_data, categories = readegi(filename, [3, 1])

    assert header["version"] == 3
    assert header["segmented"] is True
    assert header["samp_rate"] == 250
    assert header["nchan"] == 2
    assert header["segments"] == 3
    assert header["segsamps"] == 4
    assert header["eventcode"] == ["stim"]
    assert header["catname"] == ["Eyes open", "Eyes closed"]
    assert header["segment_start_times"].tolist() == [100, 300]
    np.testing.assert_array_equal(trial_data, np.concatenate((signals[:, :, 0], signals[:, :, 2]), axis=1))
    np.testing.assert_array_equal(event_data, np.concatenate((events[:, :, 0], events[:, :, 2]), axis=1))
    np.testing.assert_array_equal(categories, [1, 2])


@eeglab_test(f"{BINARY_SUITE}/pop_readegi/binary_pop_readegi_wrapperTest.m", "test_test_pop_readegi")
def test_pop_readegi_builds_an_epoched_dataset_with_events_and_categories(tmp_path: Path) -> None:
    signals, events = _segmented_values()
    filename = tmp_path / "TESTEGI.RAW"
    _write_egi_raw(
        filename,
        3,
        signals,
        events=events,
        event_codes=("stim",),
        category_indices=(1, 2, 2),
    )

    eeg, command = pop_readegi(filename, [1, 3], fileloc="", return_com=True)

    assert eeg["data"].shape == (2, 4, 2)
    np.testing.assert_array_equal(eeg["data"], signals[:, :, [0, 2]])
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 4
    assert eeg["trials"] == 2
    assert eeg["srate"] == 250
    assert eeg["xmin"] == 0
    assert eeg["xmax"] == pytest.approx(3 / 250)
    np.testing.assert_allclose(eeg["times"], [0, 4, 8, 12])
    assert [event["type"] for event in eeg["event"]] == ["stim", "stim"]
    assert [event["latency"] for event in eeg["event"]] == [3, 7]
    assert [event["epoch"] for event in eeg["event"]] == [1, 2]
    assert [event["category"] for event in eeg["event"]] == ["Eyes open", "Eyes closed"]
    assert [event["urevent"] for event in eeg["event"]] == [0, 1]
    assert eeg["history"] == command
    assert command == f"EEG = pop_readegi('{filename.as_posix()}', [1 3], [], '');"


@eeglab_test(
    f"{BINARY_SUITE}/pop_readsegegi/binary_pop_readsegegi_wrapperTest.m",
    "test_test_pop_readsegegi",
)
def test_pop_readsegegi_joins_the_numbered_continuous_series_with_valid_metadata(tmp_path: Path) -> None:
    first_signals = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]], dtype=float)
    second_signals = np.array([[6, 7, 8, 9], [16, 17, 18, 19]], dtype=float)
    first_events = np.array([[0, 0, 1, 0, 0]], dtype=float)
    second_events = np.array([[0, 1, 0, 0]], dtype=float)
    first = tmp_path / "TEST_001.RAW"
    second = tmp_path / "TEST_002.RAW"
    _write_egi_raw(first, 2, first_signals, events=first_events, event_codes=("stim",))
    _write_egi_raw(second, 2, second_signals, events=second_events, event_codes=("stim",))

    eeg, command = pop_readsegegi(second, fileloc="", return_com=True)

    np.testing.assert_array_equal(eeg["data"], np.concatenate((first_signals, second_signals), axis=1))
    assert eeg["data"].shape == (2, 9)
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 9
    assert eeg["trials"] == 1
    assert eeg["xmax"] == pytest.approx(8 / 250)
    assert [event["type"] for event in eeg["event"]] == ["stim", "stim"]
    assert [event["latency"] for event in eeg["event"]] == [3, 7]
    assert "TEST_001.RAW" in eeg["comments"]
    assert "TEST_002.RAW" in eeg["comments"]
    assert command == f"EEG = pop_readsegegi('{second.as_posix()}');"
    assert eeg["history"] == command


@pytest.mark.parametrize("version", [2, 4, 6])
def test_readegi_supports_every_continuous_sample_encoding(tmp_path: Path, version: int) -> None:
    values = np.array([[1, -2, 3], [4, 5, -6]], dtype=float)
    if version >= 4:
        values += 0.25
    filename = tmp_path / f"continuous-v{version}.raw"
    _write_egi_raw(filename, version, values, bits=0, signal_range=0)

    header, actual, event_data, categories = readegi(filename)

    assert header["version"] == version
    np.testing.assert_allclose(actual, values, rtol=1e-6)
    assert event_data.shape == (0, 3)
    assert categories.size == 0


@pytest.mark.parametrize("version", [3, 5, 7])
def test_readegi_supports_every_segmented_sample_encoding(tmp_path: Path, version: int) -> None:
    values = np.array([[[1, 5], [2, 6], [3, 7], [4, 8]]], dtype=float)
    if version >= 5:
        values += 0.125
    filename = tmp_path / f"segmented-v{version}.raw"
    _write_egi_raw(filename, version, values, category_indices=(1, 2), bits=0, signal_range=0)

    header, actual, event_data, categories = readegi(filename)

    assert header["version"] == version
    np.testing.assert_allclose(actual, np.concatenate((values[:, :, 0], values[:, :, 1]), axis=1), rtol=1e-6)
    assert event_data.shape == (0, 8)
    np.testing.assert_array_equal(categories, [1, 2])


def test_readegi_applies_ad_scaling_only_to_eeg_channels(tmp_path: Path) -> None:
    signals = np.array([[2048, -2048]], dtype=float)
    events = np.array([[0, 9]], dtype=float)
    filename = tmp_path / "scaled.raw"
    _write_egi_raw(
        filename,
        2,
        signals,
        events=events,
        event_codes=("stim",),
        bits=12,
        signal_range=1000,
    )

    _header, actual, event_data, _categories = readegi(filename)

    np.testing.assert_array_equal(actual, [[500, -500]])
    np.testing.assert_array_equal(event_data, events)


def test_readegi_forceversion_recovers_a_mislabeled_sample_encoding(tmp_path: Path) -> None:
    signals = np.array([[1.25, -2.5], [3.75, 4.5]])
    filename = tmp_path / "mislabeled.raw"
    _write_egi_raw(filename, 4, signals, header_version=2, bits=0, signal_range=0)

    header, actual, _events, _categories = readegi(filename, forceversion=4)

    assert header["file_version"] == 2
    assert header["version"] == 4
    np.testing.assert_array_equal(actual, signals)


def test_readegihdr_reads_metadata_without_samples(tmp_path: Path) -> None:
    signals = np.ones((2, 3, 2))
    filename = tmp_path / "header.raw"
    _write_egi_raw(
        filename,
        5,
        signals,
        event_codes=(),
        category_names=("A", "B"),
        category_indices=(1, 2),
        bits=0,
        signal_range=0,
    )

    header = readegihdr(filename)

    assert header["recording_time"].isoformat() == "2025-02-03T04:05:06.007000"
    assert header["sample_dtype"] == ">f4"
    assert header["sample_width"] == 4
    assert header["header_bytes"] < filename.stat().st_size


@pytest.mark.parametrize("chunks", [[0], [4], [1.5], [True], ["1"], [1, 1], [[1, 2], [3, 1]]])
def test_readegi_rejects_invalid_chunk_vectors(tmp_path: Path, chunks: object) -> None:
    filename = tmp_path / "chunks.raw"
    _write_egi_raw(filename, 2, np.ones((2, 3)))

    with pytest.raises(ValueError, match="data_chunks"):
        readegi(filename, chunks)  # type: ignore[arg-type]


def test_readegi_accepts_a_scalar_chunk_and_numpy_integer_forceversion(tmp_path: Path) -> None:
    filename = tmp_path / "scalar.raw"
    signals = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    _write_egi_raw(filename, 2, signals)

    header, data, _events, _categories = readegi(filename, 2, np.int64(2))

    assert header["version"] == 2
    np.testing.assert_array_equal(data, signals[:, 1:2])


def test_readegi_reports_truncated_sample_data(tmp_path: Path) -> None:
    filename = tmp_path / "truncated.raw"
    _write_egi_raw(filename, 2, np.ones((2, 3)))
    filename.write_bytes(filename.read_bytes()[:-1])

    with pytest.raises(ValueError, match="Unexpected end of file"):
        readegi(filename)


def test_pop_readegi_creates_time_locking_events_when_segment_event_channels_are_absent(tmp_path: Path) -> None:
    signals = np.arange(1, 13, dtype=float).reshape(2, 3, 2)
    filename = tmp_path / "categories.raw"
    _write_egi_raw(filename, 3, signals, category_indices=(2, 1))

    eeg = pop_readegi(filename, fileloc="")

    assert [event["type"] for event in eeg["event"]] == ["TLE", "TLE"]
    assert [event["latency"] for event in eeg["event"]] == [1, 4]
    assert [event["category"] for event in eeg["event"]] == ["Eyes closed", "Eyes open"]


def test_pop_readegi_places_leading_edges_on_the_first_nonzero_sample(tmp_path: Path) -> None:
    filename = tmp_path / "leading-edge.raw"
    events = np.array([[4, 0, 0, 7, 0]], dtype=float)
    _write_egi_raw(filename, 2, np.ones((2, 5)), events=events, event_codes=("stim",))

    eeg = pop_readegi(filename, fileloc="")

    assert [event["latency"] for event in eeg["event"]] == [1, 4]


def test_pop_readegi_removes_an_empty_trailing_reference_channel(tmp_path: Path) -> None:
    filename = tmp_path / "reference.raw"
    _write_egi_raw(filename, 2, np.array([[1, 2, 3], [0, 0, 0]], dtype=float))

    eeg = pop_readegi(filename, fileloc="")

    np.testing.assert_array_equal(eeg["data"], [[1, 2, 3]])
    assert eeg["nbchan"] == 1
    assert [location["labels"] for location in eeg["chanlocs"]] == ["E1"]


def test_pop_readegi_keeps_a_trailing_channel_that_becomes_nonzero_late(tmp_path: Path) -> None:
    filename = tmp_path / "late-signal.raw"
    signals = np.vstack((np.arange(12), np.r_[np.zeros(10), 1, 2]))
    _write_egi_raw(filename, 2, signals)

    eeg = pop_readegi(filename, fileloc="")

    np.testing.assert_array_equal(eeg["data"], signals)
    assert eeg["nbchan"] == 2


def test_pop_readsegegi_rejects_incompatible_series_headers(tmp_path: Path) -> None:
    first = tmp_path / "run_001.RAW"
    second = tmp_path / "run_002.RAW"
    _write_egi_raw(first, 2, np.ones((2, 3)), sample_rate=250)
    _write_egi_raw(second, 2, np.ones((2, 3)), sample_rate=500)

    with pytest.raises(ValueError, match="samp_rate"):
        pop_readsegegi(first, fileloc="")


def test_pop_readsegegi_rejects_a_gap_before_a_later_series_file(tmp_path: Path) -> None:
    first = tmp_path / "run_001.RAW"
    third = tmp_path / "run_003.RAW"
    _write_egi_raw(first, 2, np.ones((2, 3)))
    _write_egi_raw(third, 2, np.ones((2, 3)))

    with pytest.raises(ValueError, match="missing run_002.RAW before run_003.RAW"):
        pop_readsegegi(first, fileloc="")


def test_pop_readsegegi_does_not_hide_a_corrupt_next_file(tmp_path: Path) -> None:
    first = tmp_path / "run_001.RAW"
    second = tmp_path / "run_002.RAW"
    _write_egi_raw(first, 2, np.ones((2, 3)))
    second.write_bytes(b"not an EGI file")

    with pytest.raises(ValueError, match="Unexpected end of file"):
        pop_readsegegi(first, fileloc="")


def test_pop_readsegegi_requires_a_numbered_filename_and_first_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three-digit"):
        pop_readsegegi(tmp_path / "recording.RAW")
    _write_egi_raw(tmp_path / "run_002.RAW", 2, np.ones((2, 3)))
    with pytest.raises(FileNotFoundError, match="run_001.RAW"):
        pop_readsegegi(tmp_path / "run_002.RAW")

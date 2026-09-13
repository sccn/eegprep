"""Executable ports of the current EEGLAB CNT loader wrappers.

Upstream suite: sccn/eeglab_tests@ff605546f3f70868916fb8d49c007472b3257b50
EEGLAB tree: sccn/eeglab@8ac485f654d6bbb1a6acb8dc9ef3f2eaf3d409ba
CNT implementation: sccn/neuroscanio@5915f10abac2db12f07876b9cf994371973f6c49

The two upstream recordings are 14 MB and 27 MB Git LFS objects. These tests
generate specification-level CNT files so ordinary CI executes every option
path without copying those binary recordings into EEGPrep.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

import numpy as np
import pytest

import eegprep
from eegprep.functions.adminfunc.storage import MemmapData
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from eegprep.functions.popfunc.pop_loadcnt import pop_loadcnt
from eegprep.functions.sigprocfunc.loadcnt import loadcnt
from tests.eeglab_tests import eeglab_test


LOADCNT_WRAPPER = "unittesting_binary/loadcnt/binary_loadcnt_wrapperTest.m"
POP_LOADCNT_WRAPPER = "unittesting_binary/pop_loadcnt/binary_pop_loadcnt_wrapperTest.m"


def _write_cnt(
    path: Path,
    counts: np.ndarray,
    *,
    dataformat: str,
    rate: int,
    block_samples: int = 1,
    baselines: tuple[int, ...] | None = None,
    sensitivities: tuple[float, ...] | None = None,
    calibrations: tuple[float, ...] | None = None,
    events: list[dict[str, int | float]] | None = None,
    header_samples: int | None = None,
    event_type: int = 2,
) -> Path:
    counts = np.asarray(counts)
    channels, samples = counts.shape
    dtype = np.dtype("<i2" if dataformat == "int16" else "<i4")
    baselines = baselines or tuple(10 * (index + 1) for index in range(channels))
    sensitivities = sensitivities or tuple(2.0 + index for index in range(channels))
    calibrations = calibrations or tuple(102.4 + 10 * index for index in range(channels))
    events = events or []

    setup = bytearray(900)
    setup[0:12] = b"Version 3.0\x00"
    setup[121:128] = b"Subject"
    setup[205:212] = b"Session"
    struct.pack_into("<H", setup, 370, channels)
    struct.pack_into("<H", setup, 376, rate)
    struct.pack_into("<I", setup, 864, samples if header_samples is None else header_samples)
    struct.pack_into("<i", setup, 894, 1 if block_samples == 1 else block_samples * dtype.itemsize)

    channel_headers = bytearray(75 * channels)
    for index in range(channels):
        offset = index * 75
        label = f"E{index + 1}".encode()
        channel_headers[offset : offset + len(label)] = label
        struct.pack_into("<H", channel_headers, offset + 15, index + 1)
        struct.pack_into("<f", channel_headers, offset + 19, float(index))
        struct.pack_into("<f", channel_headers, offset + 23, float(-index))
        struct.pack_into("<h", channel_headers, offset + 47, baselines[index])
        struct.pack_into("<f", channel_headers, offset + 59, sensitivities[index])
        struct.pack_into("<f", channel_headers, offset + 71, calibrations[index])

    payload = bytearray()
    for start in range(0, samples, block_samples):
        stop = min(start + block_samples, samples)
        payload.extend(np.asarray(counts[:, start:stop], dtype=dtype).ravel(order="C").tobytes())

    data_offset = 900 + len(channel_headers)
    event_position = data_offset + len(payload)
    struct.pack_into("<I", setup, 886, event_position)
    event_records = bytearray()
    for event in events:
        keypad_accept = (int(event.get("accept_ev1", 0)) << 4) | int(event.get("keypad", 0))
        event_offset = (
            int(event["sample"])
            if event_type == 3
            else data_offset + (int(event["sample"]) + 1) * channels * dtype.itemsize
        )
        if event_type == 1:
            event_records.extend(
                struct.pack(
                    "<HBBi",
                    int(event.get("stimtype", 0)),
                    int(event.get("keyboard", 0)),
                    keypad_accept,
                    event_offset,
                )
            )
        else:
            event_records.extend(
                struct.pack(
                    "<HBBihhfbbb",
                    int(event.get("stimtype", 0)),
                    int(event.get("keyboard", 0)),
                    keypad_accept,
                    event_offset,
                    int(event.get("type", 0)),
                    int(event.get("code", 0)),
                    float(event.get("latency", 0)),
                    int(event.get("epochevent", 0)),
                    int(event.get("accept", 0)),
                    int(event.get("accuracy", 0)),
                )
            )
    table = struct.pack("<BII", event_type, len(event_records), 0) + event_records
    path.write_bytes(setup + channel_headers + payload + table)
    return path


@pytest.fixture
def cnt_files(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    samples = 1200
    counts16 = np.vstack(
        [
            np.arange(samples, dtype=np.int16) - 600,
            400 - np.arange(samples, dtype=np.int16),
        ]
    )
    counts32 = np.vstack(
        [
            np.arange(samples, dtype=np.int32) * 1000 - 500_000,
            300_000 - np.arange(samples, dtype=np.int32) * 250,
        ]
    )
    events = [
        {"stimtype": 7, "sample": 0, "code": 70},
        {"stimtype": 0, "keyboard": 4, "sample": 10},
        {"stimtype": 0, "keypad": 3, "sample": 20},
        {"stimtype": 99, "accept_ev1": 14, "sample": 30},
        {"stimtype": 8, "sample": 1100},
    ]
    int16_file = _write_cnt(
        tmp_path / "TEST.CNT",
        counts16,
        dataformat="int16",
        rate=1,
        block_samples=5,
        events=events,
        event_type=1,
    )
    int32_file = _write_cnt(
        tmp_path / "TEST32BIT_WITHEVENT.CNT",
        counts32,
        dataformat="int32",
        rate=1,
        events=events,
    )
    return int16_file, int32_file, counts16, counts32


@eeglab_test(LOADCNT_WRAPPER, "test_test_loadcnt")
def test_current_loadcnt_wrapper_cases(cnt_files: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path) -> None:
    int16_file, int32_file, _counts16, _counts32 = cnt_files

    case1 = loadcnt(int32_file, "dataformat", "int32")
    case2 = loadcnt(int32_file, dataformat="int32", t1=0, lddur=301)
    case3 = loadcnt(
        int32_file,
        dataformat="int32",
        t1=0,
        sample1=0,
        lddur="301",
        ldnsamples=1000,
    )
    case4 = loadcnt(int16_file, dataformat="int16")
    mapped_file = tmp_path / "map.fdt"
    case5 = loadcnt(int16_file, dataformat="int16", memmapfile=mapped_file)
    case6 = loadcnt(int16_file, dataformat="int16", t1=0, lddur=227)
    case7 = loadcnt(
        int16_file,
        dataformat="int16",
        t1=0,
        sample1=0,
        lddur="227",
        ldnsamples=1000,
    )
    case8 = loadcnt(int32_file, keystroke="on", dataformat="int32")

    assert case1["data"].shape == (2, 1200)
    assert case2["data"].shape == (2, 301)
    assert case3["data"].shape == (2, 1000)
    assert case4["data"].shape == (2, 1200)
    assert isinstance(case5["data"], MemmapData)
    np.testing.assert_array_equal(case5["data"], case4["data"])
    assert mapped_file.stat().st_size == 2 * 1200 * 4
    assert case6["data"].shape == (2, 227)
    assert case7["data"].shape == (2, 1000)
    assert case8["event"] == case1["event"]


@eeglab_test(POP_LOADCNT_WRAPPER, "test_test_pop_loadcnt")
def test_current_pop_loadcnt_wrapper_cases(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    int16_file, int32_file, _counts16, _counts32 = cnt_files

    case1 = pop_loadcnt(int32_file, dataformat="int32")
    case2 = pop_loadcnt(int32_file, dataformat="int32", t1=0, lddur=301)
    case3 = pop_loadcnt(int32_file, dataformat="int32", t1=0, sample1=0, lddur="301", ldnsamples=1000)
    case4 = pop_loadcnt(int16_file, dataformat="int16")
    case5 = pop_loadcnt(int16_file, dataformat="int16", memmapfile=tmp_path / "map.fdt")
    case6 = pop_loadcnt(int16_file, dataformat="int16", t1=0, lddur=227)
    case7 = pop_loadcnt(int16_file, dataformat="int16", t1=0, sample1=0, lddur="227", ldnsamples=1000)
    case8, command = pop_loadcnt(int32_file, keystroke="on", dataformat="int32", return_com=True)

    assert [case1["pnts"], case2["pnts"], case3["pnts"]] == [1200, 301, 1000]
    assert [case4["pnts"], case6["pnts"], case7["pnts"]] == [1200, 227, 1000]
    assert isinstance(case5["data"], MemmapData)
    assert case1["nbchan"] == case4["nbchan"] == 2
    assert case1["srate"] == case4["srate"] == 1
    assert [channel["labels"] for channel in case1["chanlocs"]] == ["E1", "E2"]
    assert [event["type"] for event in case1["event"]] == [7, "boundary", 8]
    assert [event["type"] for event in case8["event"]] == [7, "keyboard4", "keypad3", "boundary", 8]
    assert [event["latency"] for event in case8["event"]] == [1, 11, 21, 31, 1101]
    assert [event["urevent"] for event in case8["event"]] == list(range(5))
    assert "'keystroke', 'on'" in command
    assert "'dataformat', 'int32'" in command
    assert case8["history"] == command


def test_loadcnt_scales_counts_per_channel_and_honors_precision(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray],
) -> None:
    int16_file, _int32_file, counts16, _counts32 = cnt_files
    unscaled = loadcnt(int16_file, dataformat="int16", scale="off", precision="double")
    scaled = loadcnt(int16_file, dataformat="auto", precision="single")

    np.testing.assert_array_equal(unscaled["data"], counts16)
    expected = np.empty_like(counts16, dtype=np.float32)
    expected[0] = (counts16[0] - 10) * (2.0 * 102.4 / 204.8)
    expected[1] = (counts16[1] - 20) * (3.0 * 112.4 / 204.8)
    np.testing.assert_allclose(scaled["data"], expected, rtol=1e-6, atol=1e-6)
    assert unscaled["data"].dtype == np.float64
    assert scaled["data"].dtype == np.float32
    assert scaled["dataformat"] == "int16"
    assert scaled["header"]["block_samples"] == 5


def test_loadcnt_reads_int32_channel_blocks_by_byte_width(tmp_path: Path) -> None:
    counts = np.asarray(
        [
            [1, 2, 3, 4, 5, 6, 7],
            [101, 102, 103, 104, 105, 106, 107],
            [-1, -2, -3, -4, -5, -6, -7],
        ],
        dtype=np.int32,
    )
    path = _write_cnt(
        tmp_path / "blocked.cnt",
        counts,
        dataformat="int32",
        rate=250,
        block_samples=3,
    )

    loaded = loadcnt(path, dataformat="int32", scale="off", sample1=2, ldnsamples=4)

    np.testing.assert_array_equal(loaded["data"], counts[:, 2:6])
    assert loaded["header"]["channeloffset"] == 12
    assert loaded["header"]["block_samples"] == 3


def test_pop_loadcnt_crops_and_rebases_events_at_sample_boundaries(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray],
) -> None:
    _int16_file, int32_file, _counts16, _counts32 = cnt_files

    cropped = pop_loadcnt(
        int32_file,
        dataformat="int32",
        t1=200,
        sample1=10,
        ldnsamples=21,
        keystroke="on",
    )

    assert cropped["pnts"] == 21
    assert [event["type"] for event in cropped["event"]] == ["keyboard4", "keypad3", "boundary"]
    assert [event["latency"] for event in cropped["event"]] == [1, 11, 21]


def test_pop_loadcnt_normalizes_type3_global_sample_frames(tmp_path: Path) -> None:
    path = _write_cnt(
        tmp_path / "type3.cnt",
        np.arange(20, dtype=np.int32).reshape(2, 10),
        dataformat="int32",
        rate=10,
        events=[{"stimtype": 1, "sample": 0}, {"stimtype": 2, "sample": 9}],
        event_type=3,
    )

    low_level = loadcnt(path, dataformat="int32")
    eeg = pop_loadcnt(path, dataformat="int32")

    assert [event["offset"] for event in low_level["event"]] == [1, 10]
    assert [event["latency"] for event in eeg["event"]] == [1, 10]


def test_pop_fileio_routes_cnt_through_standalone_loader(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray],
) -> None:
    int16_file, _int32_file, _counts16, _counts32 = cnt_files

    eeg, command = pop_fileio(
        int16_file,
        dataformat="int16",
        blockrange=[10, 31],
        keystroke="on",
        return_com=True,
    )

    assert eeg["data"].shape == (2, 21)
    assert eeg["event"][0]["latency"] == 1
    assert eeg["event"][0]["type"] == "keyboard4"
    assert "'blockrange', [10 31]" in command
    assert eeg["history"] == command


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda path: loadcnt(path, dataformat="float32"), "dataformat"),
        (lambda path: loadcnt(path, sample1=-1), "sample1"),
        (lambda path: loadcnt(path, sample1=0.5), "sample1"),
        (lambda path: loadcnt(path, ldnsamples=0), "sample count"),
        (lambda path: loadcnt(path, memmapfile=path.with_suffix(".dat")), ".fdt"),
        (lambda path: pop_loadcnt(path, keystroke="maybe"), "keystroke"),
    ],
)
def test_cnt_loader_rejects_ambiguous_or_unsafe_options(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray],
    operation: Any,
    message: str,
) -> None:
    int16_file, _int32_file, _counts16, _counts32 = cnt_files

    with pytest.raises(ValueError, match=message):
        operation(int16_file)


def test_loadcnt_reports_truncated_headers_and_misaligned_data(tmp_path: Path) -> None:
    short = tmp_path / "short.cnt"
    short.write_bytes(b"Version 3.0")
    with pytest.raises(ValueError, match="setup header is truncated"):
        loadcnt(short)

    valid = _write_cnt(
        tmp_path / "misaligned.cnt",
        np.arange(20, dtype=np.int16).reshape(2, 10),
        dataformat="int16",
        rate=10,
    )
    payload = bytearray(valid.read_bytes())
    struct.pack_into("<I", payload, 886, struct.unpack_from("<I", payload, 886)[0] - 1)
    valid.write_bytes(payload)
    with pytest.raises(ValueError, match="exact number of channel frames"):
        loadcnt(valid, dataformat="int16")


def test_loadcnt_warns_and_truncates_a_request_at_end_of_recording(
    cnt_files: tuple[Path, Path, np.ndarray, np.ndarray],
) -> None:
    int16_file, _int32_file, counts16, _counts32 = cnt_files

    with pytest.warns(RuntimeWarning, match="only 5 remain; truncating"):
        loaded = loadcnt(int16_file, dataformat="int16", scale="off", sample1=1195, ldnsamples=20)

    np.testing.assert_array_equal(loaded["data"], counts16[:, -5:])
    assert loaded["ldnsamples"] == 5


def test_loadcnt_validates_event_table_length(tmp_path: Path) -> None:
    path = _write_cnt(
        tmp_path / "truncated-events.cnt",
        np.arange(20, dtype=np.int16).reshape(2, 10),
        dataformat="int16",
        rate=10,
        events=[{"stimtype": 1, "sample": 2}],
    )
    payload = bytearray(path.read_bytes())
    event_position = struct.unpack_from("<I", payload, 886)[0]
    struct.pack_into("<I", payload, event_position + 1, 38)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="event table is truncated"):
        loadcnt(path, dataformat="int16")


def test_loadcnt_rejects_event_offsets_between_channel_frames(tmp_path: Path) -> None:
    path = _write_cnt(
        tmp_path / "misaligned-event.cnt",
        np.arange(20, dtype=np.int16).reshape(2, 10),
        dataformat="int16",
        rate=10,
        events=[{"stimtype": 1, "sample": 2}],
    )
    payload = bytearray(path.read_bytes())
    event_position = struct.unpack_from("<I", payload, 886)[0]
    offset_position = event_position + 9 + 4
    stored_offset = struct.unpack_from("<i", payload, offset_position)[0]
    struct.pack_into("<i", payload, offset_position, stored_offset + 1)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="not aligned"):
        loadcnt(path, dataformat="int16")


def test_loadcnt_auto_uses_explicitly_warned_int16_fallback(tmp_path: Path) -> None:
    counts = np.arange(20, dtype=np.int16).reshape(2, 10)
    path = _write_cnt(
        tmp_path / "ambiguous.cnt",
        counts,
        dataformat="int16",
        rate=10,
        header_samples=0,
    )

    with pytest.warns(RuntimeWarning, match="sample width is ambiguous"):
        loaded = loadcnt(path, scale="off")

    assert loaded["dataformat"] == "int16"
    np.testing.assert_array_equal(loaded["data"], counts)


def test_cnt_public_exports_are_available() -> None:
    assert eegprep.loadcnt is loadcnt
    assert eegprep.pop_loadcnt is pop_loadcnt

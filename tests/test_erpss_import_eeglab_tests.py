"""Current ``eeglab_tests`` coverage for ERPSS recording import."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from eegprep import pop_read_erpss, read_erpss
from tests.eeglab_tests import eeglab_test


ERPSS_SOURCE = "unittesting_binary/pop_read_erpss/binary_pop_read_erpss_wrapperTest.m"


@eeglab_test(ERPSS_SOURCE, "test_test_pop_read_erpss")
def test_pop_read_erpss_imports_both_upstream_compressed_recordings(tmp_path: Path) -> None:
    """Strengthen the upstream test, which only calls the importer without assertions."""
    first_blocks = [
        np.array(
            [
                [100, 102, 82, 282, -18, 1982],
                [-100, -104, -70, -300, -301, -1800],
            ],
            dtype=np.int16,
        ),
        np.array([[50, 53, 21, 277], [-50, -46, -280, -25]], dtype=np.int16),
    ]
    first = tmp_path / "ERPSSTESTCOMP.RAW"
    _write_erpss(
        first,
        first_blocks,
        compressed=True,
        byteorder="little",
        labels=["Cz", "Pz"],
        metadata=(1000, 1_000_000, 1000, 2048, 12, 2),
        events=[[(0, 1, 7), (5, 2, 9)], [(1, 3, 11)]],
    )

    data, events, header = read_erpss(first)

    np.testing.assert_array_equal(data, np.concatenate(first_blocks, axis=1) * 0.5)
    assert events == [
        {"sample_offset": 1, "event_code": 7, "cond_code": 1},
        {"sample_offset": 6, "event_code": 9, "cond_code": 2},
        {"sample_offset": 8, "event_code": 11, "cond_code": 3},
    ]
    assert header["nchans"] == 2
    assert header["nframes"] == 10
    assert header["nblocks"] == 2
    assert header["compressed"] is True
    assert header["byteorder"] == "little"
    assert header["chanlabels"] == ["Cz", "Pz"]
    assert header["srate"] == 500
    assert header["rescaleuv"] == 0.5

    eeg, command = pop_read_erpss(first, 999, return_com=True)

    np.testing.assert_array_equal(eeg["data"], data)
    assert eeg["srate"] == 500
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 10
    assert eeg["trials"] == 1
    assert eeg["xmin"] == 0
    assert eeg["xmax"] == pytest.approx(9 / 500)
    assert [channel["labels"] for channel in eeg["chanlocs"]] == ["Cz", "Pz"]
    assert [event["type"] for event in eeg["event"]] == [7, 9, 11]
    assert [event["latency"] for event in eeg["event"]] == [1, 6, 8]
    assert [event["urevent"] for event in eeg["event"]] == [0, 1, 2]
    assert eeg["setname"] == "ERPSS data"
    assert eeg["filename"] == first.name
    assert eeg["filepath"] == str(tmp_path)
    assert eeg["history"] == command
    assert command == f"EEG = pop_read_erpss('{first.as_posix()}', 500);"

    second = tmp_path / "ERPSSCOMPRESSED.RAW"
    second_data = np.array([[12, 8, 42], [-10, 24, -206]], dtype=np.int16)
    _write_erpss(
        second,
        [second_data],
        compressed=True,
        byteorder="big",
        labels=["F3", "F4"],
        events=[[(2, 4, 128)]],
    )

    second_eeg = pop_read_erpss(second, 500)

    np.testing.assert_array_equal(second_eeg["data"], second_data)
    assert second_eeg["srate"] == 500
    assert [event["type"] for event in second_eeg["event"]] == [128]
    assert [event["latency"] for event in second_eeg["event"]] == [3]


def test_read_erpss_reads_uncompressed_big_endian_samples_and_header_rate(tmp_path: Path) -> None:
    recording = tmp_path / "uncompressed.rdf"
    expected = np.array(
        [[32767, -32768, 123], [-456, 789, -1024], [9, 8, 7]],
        dtype=np.int16,
    )
    _write_erpss(
        recording,
        [expected],
        compressed=False,
        byteorder="big",
        labels=["A1", "A2", "EOG"],
        sample_interval=4000,
    )

    data, events, header = read_erpss(recording)
    eeg = pop_read_erpss(recording)

    np.testing.assert_array_equal(data, expected)
    assert events == []
    assert header["compressed"] is False
    assert header["byteorder"] == "big"
    assert header["srate"] == 250
    assert eeg["srate"] == 250
    assert [channel["labels"] for channel in eeg["chanlocs"]] == ["A1", "A2", "EOG"]


def test_read_erpss_rejects_corrupt_and_truncated_recordings(tmp_path: Path) -> None:
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    recording = tmp_path / "recording.raw"
    _write_erpss(recording, [data], compressed=True, byteorder="little", labels=["A", "B"])

    invalid_tag = tmp_path / "invalid-tag.raw"
    invalid_tag.write_bytes(b"bad!" + recording.read_bytes()[4:])
    with pytest.raises(ValueError, match="byte-order tag"):
        read_erpss(invalid_tag)

    truncated = tmp_path / "truncated.raw"
    truncated.write_bytes(recording.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated sample data"):
        read_erpss(truncated)

    invalid_delta = tmp_path / "invalid-delta.raw"
    invalid_bytes = bytearray(recording.read_bytes())
    invalid_bytes[4608:4610] = b"\x00\x00"
    invalid_delta.write_bytes(invalid_bytes)
    with pytest.raises(ValueError, match="starts with a delta"):
        read_erpss(invalid_delta)

    outside_event = tmp_path / "outside-event.raw"
    _write_erpss(
        outside_event,
        [data],
        compressed=False,
        byteorder="little",
        labels=["A", "B"],
        events=[[(3, 0, 1)]],
    )
    with pytest.raises(ValueError, match="outside the block"):
        read_erpss(outside_event)


def test_pop_read_erpss_requires_a_rate_when_the_header_has_none(tmp_path: Path) -> None:
    recording = tmp_path / "unknown-rate.raw"
    _write_erpss(
        recording,
        [np.array([[1, 2]], dtype=np.int16)],
        compressed=False,
        byteorder="little",
        labels=["Cz"],
    )

    with pytest.raises(ValueError, match="sampling rate"):
        pop_read_erpss(recording)
    with pytest.raises(ValueError, match="finite positive"):
        pop_read_erpss(recording, 0)


def _write_erpss(
    path: Path,
    blocks: list[np.ndarray],
    *,
    compressed: bool,
    byteorder: str,
    labels: list[str],
    metadata: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    events: list[list[tuple[int, int, int]]] | None = None,
    sample_interval: int = 0,
) -> None:
    prefix = "<" if byteorder == "little" else ">"
    file_tag = b"\x55\xaa\xb0\x00" if byteorder == "little" else b"\xaa\x55\x00\xb0"
    block_tag = 0x00F0AA55 if byteorder == "little" else 0xAA5500F0
    nchans = blocks[0].shape[0]
    assert all(block.ndim == 2 and block.shape[0] == nchans for block in blocks)
    assert len(labels) == nchans
    header = bytearray(4096)
    header[:4] = file_tag
    struct.pack_into(f"{prefix}H", header, 4, 1 if compressed else 0)
    struct.pack_into(f"{prefix}H", header, 6, nchans)
    struct.pack_into(f"{prefix}H", header, 552, sample_interval)
    for index, label in enumerate(labels):
        encoded = label.encode("latin-1")
        assert len(encoded) <= 8
        start = 580 + index * 8
        header[start : start + len(encoded)] = encoded
    output = bytearray(header)
    block_events = events or [[] for _ in blocks]
    assert len(block_events) == len(blocks)
    for block, current_events in zip(blocks, block_events):
        payload = _compress(block, prefix) if compressed else block.T.astype(f"{prefix}i2").tobytes()
        raw_block_header = bytearray(512)
        struct.pack_into(
            f"{prefix}I10H",
            raw_block_header,
            0,
            block_tag,
            1,
            nchans,
            6,
            block.shape[1],
            0,
            1,
            0,
            0,
            len(current_events),
            len(payload) // 2 if compressed else 0,
        )
        struct.pack_into(f"{prefix}6I", raw_block_header, 24, *metadata)
        for index, event in enumerate(current_events):
            struct.pack_into(f"{prefix}BBH", raw_block_header, 72 + index * 4, *event)
        output.extend(raw_block_header)
        output.extend(payload)
    path.write_bytes(output)


def _compress(data: np.ndarray, prefix: str) -> bytes:
    bits: list[int] = []

    def append(value: int, width: int) -> None:
        bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))

    nchans = data.shape[0]
    flat = data.T.astype(np.int32).reshape(-1)
    for index, sample in enumerate(flat):
        previous = int(flat[index - nchans]) if index >= nchans else None
        difference = int(sample) - previous if previous is not None else None
        if difference is not None and -4 <= difference <= 3:
            append(difference & 0b111, 4)
        elif difference is not None and -32 <= difference <= 31:
            encoded = difference & 0b11111
            code = 0b1000 | ((difference < 0) << 1) | (encoded >> 4)
            append(code, 4)
            append(encoded, 4)
        elif difference is not None and -256 <= difference <= 255:
            append(0b1101 if difference < 0 else 0b1100, 4)
            append(difference & 0xFF, 8)
        else:
            assert -2048 <= sample <= 2047
            append(0b1110, 4)
            append(int(sample) & 0xFFF, 12)
    bits.extend([0] * (-len(bits) % 16))
    words = []
    for start in range(0, len(bits), 16):
        word = 0
        for bit in bits[start : start + 16]:
            word = (word << 1) | bit
        words.append(word)
    return struct.pack(f"{prefix}{len(words)}H", *words)

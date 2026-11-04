from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from eegprep import openbdf, readbdf
from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from tests.eeglab_tests import eeglab_test


OPENBDF_SUITE = "unittesting_sigprocfunc/openbdf/sigprocfunc_openbdf_wrapperTest.m"
READBDF_SUITE = "unittesting_binary/readbdf/binary_readbdf_wrapperTest.m"


def _skip_without_openbdf(request):
    """The Biosig extension zip does not ship openbdf."""
    if request.config.getoption("--eeglab-backend") != "matlab":
        return
    found = request.getfixturevalue("eeglab_matlab_engine").which("openbdf")
    if isinstance(found, str) and found.strip():
        return
    pytest.skip("Biosig plugin does not provide openbdf")


def _field(value: str | int | float | None, width: int) -> bytes:
    text = "" if value is None else str(value)
    encoded = text.encode("ascii")
    if len(encoded) > width:
        raise ValueError(f"{text!r} does not fit a {width}-byte BDF field")
    return encoded.ljust(width)


def _encode_signed_24(values: np.ndarray) -> bytes:
    unsigned = np.asarray(values, dtype=np.int64) & 0xFFFFFF
    packed = np.empty((unsigned.size, 3), dtype=np.uint8)
    packed[:, 0] = unsigned & 0xFF
    packed[:, 1] = (unsigned >> 8) & 0xFF
    packed[:, 2] = (unsigned >> 16) & 0xFF
    return packed.tobytes()


def _write_bdf(
    path: Path,
    *,
    labels: Sequence[str],
    samples_per_record: Sequence[int],
    records: Sequence[Sequence[np.ndarray]] = (),
    declared_records: int | None = None,
    physical_minimum: Sequence[int | float | None] | None = None,
    physical_maximum: Sequence[int | float | None] | None = None,
    digital_minimum: Sequence[int | float | None] | None = None,
    digital_maximum: Sequence[int | float | None] | None = None,
) -> None:
    signal_count = len(labels)
    if len(samples_per_record) != signal_count:
        raise ValueError("samples_per_record must match labels")
    physical_minimum = physical_minimum or [-262144] * signal_count
    physical_maximum = physical_maximum or [262144] * signal_count
    digital_minimum = digital_minimum or [-8388608] * signal_count
    digital_maximum = digital_maximum or [8388607] * signal_count
    record_count = len(records) if declared_records is None else declared_records
    head_length = 256 + signal_count * 256
    fixed = b"".join(
        (
            b"\xffBIOSEMI",
            _field("Local subject identification", 80),
            _field("Local recording identification", 80),
            _field("12.03.07", 8),
            _field("15.35.27", 8),
            _field(head_length, 8),
            _field("24BIT", 44),
            _field(record_count, 8),
            _field(2, 8),
            _field(signal_count, 4),
        )
    )
    signal_header = b"".join(
        (
            b"".join(_field(value, 16) for value in labels),
            b"".join(_field("Active Electrode, pin type", 80) for _ in labels),
            b"".join(_field("uV", 8) for _ in labels),
            b"".join(_field(value, 8) for value in physical_minimum),
            b"".join(_field(value, 8) for value in physical_maximum),
            b"".join(_field(value, 8) for value in digital_minimum),
            b"".join(_field(value, 8) for value in digital_maximum),
            b"".join(_field("HP: DC; LP: 113 Hz", 80) for _ in labels),
            b"".join(_field(value, 8) for value in samples_per_record),
            b"".join(_field("", 32) for _ in labels),
        )
    )
    data = bytearray()
    for record in records:
        if len(record) != signal_count:
            raise ValueError("each record must contain every signal")
        for values, count in zip(record, samples_per_record, strict=True):
            array = np.asarray(values)
            if array.shape != (count,):
                raise ValueError("record signal length does not match samples_per_record")
            data.extend(_encode_signed_24(array))
    path.write_bytes(fixed + signal_header + data)


def test_pop_biosig_drops_bdf_status_and_reads_trigger_edges(tmp_path: Path) -> None:
    path = tmp_path / "status.bdf"
    labels = ["A1", "A2", "Status"]
    eeg_a = np.arange(8, dtype=int)
    eeg_b = np.arange(8, dtype=int) + 20
    status = np.array([0, 0, 5, 5, 0, 0, 7, 0], dtype=int)
    _write_bdf(
        path,
        labels=labels,
        samples_per_record=[8, 8, 8],
        records=[[eeg_a, eeg_b, status]],
        physical_minimum=[-262144, -262144, -8388608],
        physical_maximum=[262144, 262144, 8388607],
        digital_minimum=[-8388608, -8388608, -8388608],
        digital_maximum=[8388607, 8388607, 8388607],
    )

    eeg = pop_biosig(path)
    opened = openbdf(path)
    calibrated, _raw = readbdf(opened, [1])

    assert [loc["labels"] for loc in eeg["chanlocs"]] == ["A1", "A2"]
    assert eeg["nbchan"] == 2
    assert eeg["data"].shape == (2, 8)
    np.testing.assert_allclose(eeg["data"], calibrated["Record"][:2])
    assert [(event["type"], event["latency"]) for event in eeg["event"]] == [(5, 3), (7, 7)]
    _eeg, command = pop_biosig(path, return_com=True)
    assert "bdfeventmode" not in command
    assert "rmeventchan" not in command
    assert "importevent" not in command


def _write_status_bdf(
    path: Path,
    status_records: Sequence[np.ndarray],
    *,
    declared_records: int | None = None,
) -> None:
    records = [np.asarray(record, dtype=int) for record in status_records]
    length = int(records[0].shape[0])
    labels = ["A1", "A2", "Status"]
    _write_bdf(
        path,
        labels=labels,
        samples_per_record=[length, length, length],
        records=[[np.zeros(length, dtype=int), np.zeros(length, dtype=int), record] for record in records],
        declared_records=declared_records,
        physical_minimum=[-262144, -262144, -8388608],
        physical_maximum=[262144, 262144, 8388607],
        digital_minimum=[-8388608, -8388608, -8388608],
        digital_maximum=[8388607, 8388607, 8388607],
    )


def _event_pairs(eeg: dict) -> list[tuple[object, object]]:
    return [(event["type"], event["latency"]) for event in eeg["event"]]


def test_pop_biosig_honors_status_import_and_removal(tmp_path: Path) -> None:
    path = tmp_path / "options.bdf"
    _write_status_bdf(path, [np.array([0, 0, 5, 5, 0, 0, 7, 0])])

    kept, command = pop_biosig(path, rmeventchan="off", return_com=True)
    assert [loc["labels"] for loc in kept["chanlocs"]] == ["A1", "A2", "Status"]
    assert kept["nbchan"] == 3
    assert kept["data"].shape == (3, 8)
    assert _event_pairs(kept) == [(5, 3), (7, 7)]
    assert "'rmeventchan', 'off'" in command

    without_events = pop_biosig(path, importevent="off")
    assert [loc["labels"] for loc in without_events["chanlocs"]] == ["A1", "A2"]
    assert _event_pairs(without_events) == []

    kept_without_events = pop_biosig(path, importevent="off", rmeventchan="off")
    assert [loc["labels"] for loc in kept_without_events["chanlocs"]] == ["A1", "A2", "Status"]
    assert _event_pairs(kept_without_events) == []


def test_pop_biosig_rejects_unsupported_bdf_event_options(tmp_path: Path) -> None:
    path = tmp_path / "options.bdf"
    _write_status_bdf(path, [np.array([0, 5, 0, 0])])

    with pytest.raises(ValueError, match="bdfeventmode"):
        pop_biosig(path, bdfeventmode=9)
    with pytest.raises(ValueError, match="bdfeventmode"):
        pop_biosig(path, bdfeventmode=4.5)
    with pytest.raises(ValueError, match="bdfeventmode"):
        pop_biosig(path, bdfeventmode="4")
    with pytest.raises(ValueError, match="rmeventchan"):
        pop_biosig(path, rmeventchan="maybe")
    with pytest.raises(ValueError, match="importevent"):
        pop_biosig(path, importevent="yes")


def test_pop_biosig_decodes_each_supported_bdfeventmode(tmp_path: Path) -> None:
    path = tmp_path / "modes.bdf"
    _write_status_bdf(path, [np.array([5, 0, 0x105, 0x10005, 4])])
    expected = {
        1: [(5, 1), (32773, 2), (5, 3), (32766, 4), (32773, 4), (65534, 5), (32773, 5)],
        2: [(5, 1), (261, 3)],
        3: [(5, 1), (32773, 2), (261, 3), (33029, 4), (4, 5)],
        4: [(5, 1), (261, 3), (4, 5)],
        5: [(5, 1), (32773, 2), (5, 3), (32773, 4), (4, 5)],
        6: [(5, 1), (5, 3), (4, 5)],
        7: [(1, 2), (3, 2), (32769, 3), (32771, 3), (9, 3), (32777, 4), (1, 5)],
        8: [(1, 1), (3, 1), (1, 3), (3, 3), (9, 3)],
        99: [(5, 1), (32773, 2), (261, 3), (33029, 4), (32773, 5)],
    }

    for mode, pairs in expected.items():
        eeg, command = pop_biosig(path, bdfeventmode=mode, return_com=True)
        assert _event_pairs(eeg) == pairs
        assert all(type(event["type"]) is int for event in eeg["event"])
        assert f"'bdfeventmode', {mode}" in command
        assert [loc["labels"] for loc in eeg["chanlocs"]] == ["A1", "A2"]
    assert _event_pairs(pop_biosig(path, bdfeventmode=4.0)) == expected[4]

    high_bit = tmp_path / "high-bit.bdf"
    _write_status_bdf(high_bit, [np.array([0, 0x8001, 0])])
    assert _event_pairs(pop_biosig(high_bit, bdfeventmode=3)) == [(0x8001, 2), (0x8001 + 0x8000, 3)]
    assert _event_pairs(pop_biosig(high_bit, bdfeventmode=99)) == [(0x8001, 2), (0x8001 | 0x8000, 3)]


@pytest.mark.filterwarnings("ignore:Number of records from the header does not match the file size:RuntimeWarning")
def test_pop_biosig_reads_only_blockrange_status_records(tmp_path: Path) -> None:
    early = tmp_path / "early.bdf"
    _write_status_bdf(early, [np.array([0, 5, 0, 0])], declared_records=6)
    early_eeg = pop_biosig(early, blockrange=[0, 2])
    assert _event_pairs(early_eeg) == [(5, 2)]

    window = tmp_path / "window.bdf"
    _write_status_bdf(
        window,
        [np.array([0, 0, 1, 1]), np.array([1, 0, 4, 0])],
        declared_records=6,
    )
    window_eeg = pop_biosig(window, blockrange=[2, 4])
    assert _event_pairs(window_eeg) == [(4, 3)]

    typed = tmp_path / "typed.bdf"
    _write_status_bdf(typed, [np.array([0, 5, 0, 0]), np.array([0, 7, 0, 9])])
    assert _event_pairs(pop_biosig(typed, blockrange=[2, 4])) == [(7, 2), (9, 4)]

    anchored = tmp_path / "anchored.bdf"
    _write_status_bdf(
        anchored,
        [np.ones(4, dtype=int), np.zeros(4, dtype=int), np.array([0, 1, 1, 0])],
        declared_records=6,
    )
    anchored_eeg = pop_biosig(anchored, blockrange=[4, 6], bdfeventmode=7)
    assert _event_pairs(anchored_eeg) == [(32769, 2), (1, 4)]


def test_pop_biosig_short_status_window_stays_channel_major(tmp_path: Path) -> None:
    path = tmp_path / "short-status.bdf"
    samples = 4
    labels = [f"A{index}" for index in range(1, 6)] + ["Status"]
    rows = [np.arange(samples, dtype=int) + 10 * index for index in range(5)]
    status = np.array([0, 5, 5, 0], dtype=int)
    _write_bdf(
        path,
        labels=labels,
        samples_per_record=[samples] * len(labels),
        records=[rows + [status]],
        physical_minimum=[-262144] * 5 + [-8388608],
        physical_maximum=[262144] * 5 + [8388607],
        digital_minimum=[-8388608] * len(labels),
        digital_maximum=[8388607] * len(labels),
    )
    opened = openbdf(path)
    calibrated, _raw = readbdf(opened, [1])

    window = pop_biosig(path, blockrange=[0, 1])
    entire = pop_biosig(path)
    via_fileio = pop_fileio(path, blockrange=[0, 1])

    assert window["data"].shape == (5, 2)
    assert window["nbchan"] == 5
    assert window["pnts"] == 2
    assert window["nbchan"] > window["pnts"]
    assert [loc["labels"] for loc in window["chanlocs"]] == labels[:-1]
    np.testing.assert_allclose(window["data"], calibrated["Record"][:5, :2])
    assert _event_pairs(window) == [(5, 2)]
    assert entire["data"].shape == (5, samples)
    assert entire["nbchan"] > entire["pnts"]
    np.testing.assert_allclose(entire["data"], calibrated["Record"][:5])
    assert via_fileio["data"].shape == (len(labels), 2)
    np.testing.assert_allclose(via_fileio["data"][:-1], window["data"])


def test_openbdf_general_header_matches_current_suite(tmp_path: Path) -> None:
    path = tmp_path / "test.bdf"
    labels = [f"A{index}" for index in range(1, 17)] + ["Status"]
    samples = [256] * 17
    records = [[np.zeros(256, dtype=int) for _ in labels] for _ in range(30)]
    _write_bdf(path, labels=labels, samples_per_record=samples, records=records)

    dataset, raw_header = openbdf(path, return_header=True)
    head = dataset["Head"]

    assert dataset["MX"] == {"ReRef": 1}
    assert len(raw_header) == 256
    assert raw_header.startswith("\xffBIOSEMI")
    assert head["VERSION"] == "\xffBIOSEMI"
    assert head["PID"] == "Local subject identification"
    assert head["RID"] == "Local recording identification"
    assert head["T0"] == [7, 3, 12, 15, 35, 27]
    assert head["StartDateTime"].isoformat() == "2007-03-12T15:35:27"
    assert head["HeadLen"] == 4608
    assert head["NRec"] == 30
    assert head["Dur"] == 2
    assert head["NS"] == 17
    assert head["Label"] == labels
    np.testing.assert_array_equal(head["SPR"], samples)
    np.testing.assert_array_equal(head["SampleRate"], np.full(17, 128.0))
    np.testing.assert_array_equal(head["Chan_Select"], np.ones(17, dtype=bool))
    assert head["ChanTyp"] == "N" * 17
    assert head["AS"]["spb"] == 256 * 17
    np.testing.assert_array_equal(head["AS"]["IDX2"], np.arange(1, 256 * 17 + 1))
    np.testing.assert_allclose(head["Cal"][:16], np.full(16, 524288 / 16777215))
    assert head["Cal"][-1] == pytest.approx(524288 / 16777215)
    assert head["FILE"]["FID"] is None
    assert head["FILE"]["POS"] == 4608
    assert head["FileName"] == str(path.resolve())


def test_openbdf_unknown_record_count_uses_three_byte_samples(tmp_path: Path) -> None:
    path = tmp_path / "test_unknown_record_size.bdf"
    labels = ["A1", "A2"]
    records = [[np.arange(4), np.arange(4) + record] for record in range(60)]
    _write_bdf(path, labels=labels, samples_per_record=[4, 4], records=records, declared_records=-1)

    assert openbdf(path)["Head"]["NRec"] == 60


def test_readbdf_decodes_selected_records_and_calibrates_them(tmp_path: Path) -> None:
    path = tmp_path / "test.bdf"
    labels = [f"A{index}" for index in range(1, 17)] + ["Status"]
    digital_minimum = [-8388608] * 17
    digital_maximum = [8388607] * 17
    physical_minimum = [-262144] * 16 + [-8388608]
    physical_maximum = [262144] * 16 + [8388607]
    records: list[list[np.ndarray]] = []
    for record_index in range(1, 5):
        record = [record_index * 100000 + channel * 1000 + np.arange(256) - 300000 for channel in range(17)]
        records.append(record)
    records[1][0][3] = digital_maximum[0]
    records[3][1][5] = digital_minimum[1]
    _write_bdf(
        path,
        labels=labels,
        samples_per_record=[256] * 17,
        records=records,
        physical_minimum=physical_minimum,
        physical_maximum=physical_maximum,
        digital_minimum=digital_minimum,
        digital_maximum=digital_maximum,
    )
    opened = openbdf(path)

    result, last_raw = readbdf(opened, [1, 2, 4])

    expected_raw = np.column_stack(records[3]).astype(float)
    expected_digital = np.concatenate([np.column_stack(records[index]) for index in (0, 1, 3)], axis=0)
    expected_physical = expected_digital * opened["Head"]["Cal"] + opened["Head"]["Off"]
    np.testing.assert_array_equal(last_raw, expected_raw)
    np.testing.assert_allclose(result["Record"], expected_physical.T)
    np.testing.assert_array_equal(result["Idx"], [1, 2, 4])
    assert result["Record"].shape == (17, 256 * 3)
    assert result["Valid"].shape == (1, 256 * 3)
    invalid = np.flatnonzero(result["Valid"][0] == 0)
    np.testing.assert_array_equal(invalid, [256 + 3, 512 + 5])
    raw_result, _ = readbdf(opened, [1, 2, 4], mode=1)
    np.testing.assert_array_equal(raw_result["Record"], expected_digital.T)
    assert "Record" not in opened


def test_readbdf_preserves_variable_rate_padding_and_compact_modes(tmp_path: Path) -> None:
    path = tmp_path / "variable.bdf"
    records = [
        [np.asarray([1, 2, 3, 4]), np.asarray([10, 11])],
        [np.asarray([5, 6, 7, 8]), np.asarray([12, 13])],
    ]
    _write_bdf(
        path,
        labels=["EEG", "Aux"],
        samples_per_record=[4, 2],
        records=records,
        physical_minimum=[-8388608, -8388608],
        physical_maximum=[8388607, 8388607],
    )
    opened = openbdf(path)

    padded, _ = readbdf(opened, [1, 2], mode=1)
    compact, _ = readbdf(opened, [1, 2], mode=3)

    np.testing.assert_array_equal(padded["Record"][0], [1, 2, 3, 4, 5, 6, 7, 8])
    np.testing.assert_array_equal(padded["Record"][1], [10, 11, np.nan, np.nan, 12, 13, np.nan, np.nan])
    np.testing.assert_array_equal(compact["Record"][1], [10, 11, 12, 13, 0, 0, 0, 0])


@eeglab_test(OPENBDF_SUITE, "test_pass_general")
def test_upstream_openbdf_original_general_header(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    # The source constructs an expected header but does not compare it.
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test.bdf"
    eeglab_backend("openbdf", str(path), nargout=2)


@eeglab_test(OPENBDF_SUITE, "test_pass_chan_types")
def test_upstream_openbdf_original_channel_types(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_chan_types.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    assert data["Head"]["ChanTyp"] == "N CCEOM"


@eeglab_test(OPENBDF_SUITE, "test_pass_dig_min_larger_max")
def test_upstream_openbdf_original_reversed_digital_limits(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_dig_min_larger_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["Cal"][:2], np.ones((2, 1)))
    np.testing.assert_array_equal(data["Head"]["Off"][:2], np.zeros((2, 1)))


@eeglab_test(OPENBDF_SUITE, "test_pass_invalid_dig_min_max")
def test_upstream_openbdf_original_invalid_digital_limits(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_invalid_dig_min_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["DigMin"], np.full((7, 1), -32768.0))
    np.testing.assert_array_equal(data["Head"]["DigMax"], np.full((7, 1), 32767.0))


@eeglab_test(OPENBDF_SUITE, "test_pass_invalid_phys_min_max")
def test_upstream_openbdf_original_invalid_physical_limits(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_invalid_phys_min_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["PhysMin"], data["Head"]["DigMin"])
    np.testing.assert_array_equal(data["Head"]["PhysMax"], data["Head"]["DigMax"])


@eeglab_test(OPENBDF_SUITE, "test_pass_phys_min_larger_max")
def test_upstream_openbdf_original_reversed_physical_limits(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_phys_min_larger_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["PhysMin"], data["Head"]["DigMin"])
    np.testing.assert_array_equal(data["Head"]["PhysMax"], data["Head"]["DigMax"])


@eeglab_test(READBDF_SUITE, "test_pass_general")
def test_upstream_readbdf_original_record_selection(eeglab_backend, request, eeglab_suite_root):
    _skip_without_openbdf(request)
    path = eeglab_suite_root / "unittesting_binary/readbdf/test.bdf"
    bdf = eeglab_backend("openbdf", str(path))
    records = np.array([[1.0, 2.0, 4.0]])
    data, signal = eeglab_backend("readbdf", bdf, records, nargout=2)
    assert data["Record"].shape == (17, 256 * 3)
    assert data["Valid"].shape == (1, 256 * 3)
    np.testing.assert_array_equal(data["Idx"], records)
    assert signal.shape == (256, 17)

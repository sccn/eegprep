from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from eegprep import openbdf, readbdf
from tests.eeglab_tests import eeglab_test


OPENBDF_SUITE = "unittesting_sigprocfunc/openbdf/sigprocfunc_openbdf_wrapperTest.m"
READBDF_SUITE = "unittesting_binary/readbdf/binary_readbdf_wrapperTest.m"


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


def test_openbdf_channel_types_match_current_suite(tmp_path: Path) -> None:
    path = tmp_path / "test_chan_types.bdf"
    _write_bdf(
        path,
        labels=["A1", "A2", "ECG", "EKG", "EEG", "EOG", "EMG"],
        samples_per_record=[256, 200, 256, 256, 256, 256, 256],
    )

    assert openbdf(path)["Head"]["ChanTyp"] == "N CCEOM"


def test_openbdf_invalid_digital_order_disables_calibration(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "test_dig_min_larger_max.bdf"
    _write_bdf(
        path,
        labels=["A1", "A2", "A3"],
        samples_per_record=[4, 4, 4],
        digital_minimum=[8388608, 8388608, -8388608],
        digital_maximum=[8388607, 8388607, 8388607],
    )

    head = openbdf(path)["Head"]

    np.testing.assert_array_equal(head["Cal"][:2], [1, 1])
    np.testing.assert_array_equal(head["Off"][:2], [0, 0])
    assert "digital minimum is not smaller" in caplog.text


def test_openbdf_missing_digital_limits_use_legacy_fallback(tmp_path: Path) -> None:
    path = tmp_path / "test_invalid_dig_min_max.bdf"
    _write_bdf(
        path,
        labels=[f"A{index}" for index in range(7)],
        samples_per_record=[4] * 7,
        digital_minimum=[None] * 7,
        digital_maximum=[None] * 7,
    )

    head = openbdf(path)["Head"]

    np.testing.assert_array_equal(head["DigMin"], np.full(7, -32768))
    np.testing.assert_array_equal(head["DigMax"], np.full(7, 32767))


def test_openbdf_missing_physical_limits_use_digital_limits(tmp_path: Path) -> None:
    path = tmp_path / "test_invalid_phys_min_max.bdf"
    _write_bdf(
        path,
        labels=[f"A{index}" for index in range(7)],
        samples_per_record=[4] * 7,
        physical_minimum=[None] * 7,
        physical_maximum=[None] * 7,
    )

    head = openbdf(path)["Head"]

    np.testing.assert_array_equal(head["PhysMin"], head["DigMin"])
    np.testing.assert_array_equal(head["PhysMax"], head["DigMax"])


def test_openbdf_reversed_physical_limits_use_digital_limits(tmp_path: Path) -> None:
    path = tmp_path / "test_phys_min_larger_max.bdf"
    _write_bdf(
        path,
        labels=[f"A{index}" for index in range(7)],
        samples_per_record=[4] * 7,
        physical_minimum=[2, 2, -2, -2, -2, -2, -2],
        physical_maximum=[1, 1, 2, 2, 2, 2, 2],
    )

    head = openbdf(path)["Head"]

    np.testing.assert_array_equal(head["PhysMin"], head["DigMin"])
    np.testing.assert_array_equal(head["PhysMax"], head["DigMax"])


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


def test_readbdf_rejects_fractional_out_of_range_and_truncated_records(tmp_path: Path) -> None:
    path = tmp_path / "short.bdf"
    _write_bdf(
        path,
        labels=["A1"],
        samples_per_record=[4],
        records=[[np.arange(4)]],
    )
    opened = openbdf(path)

    with pytest.raises(ValueError, match="integers"):
        readbdf(opened, [1.5])
    with pytest.raises(IndexError, match="between 1 and 1"):
        readbdf(opened, [2])
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="incomplete"):
        readbdf(opened, [1])


@eeglab_test(OPENBDF_SUITE, "test_pass_general")
def test_upstream_openbdf_original_general_header(eeglab_backend, eeglab_suite_root):
    # The source constructs an expected header but does not compare it.
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test.bdf"
    eeglab_backend("openbdf", str(path), nargout=2)


@eeglab_test(OPENBDF_SUITE, "test_pass_chan_types")
def test_upstream_openbdf_original_channel_types(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_chan_types.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    assert data["Head"]["ChanTyp"] == "N CCEOM"


@eeglab_test(OPENBDF_SUITE, "test_pass_dig_min_larger_max")
def test_upstream_openbdf_original_reversed_digital_limits(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_dig_min_larger_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["Cal"][:2], np.ones((2, 1)))
    np.testing.assert_array_equal(data["Head"]["Off"][:2], np.zeros((2, 1)))


@eeglab_test(OPENBDF_SUITE, "test_pass_invalid_dig_min_max")
def test_upstream_openbdf_original_invalid_digital_limits(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_invalid_dig_min_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["DigMin"], np.full((7, 1), -32768.0))
    np.testing.assert_array_equal(data["Head"]["DigMax"], np.full((7, 1), 32767.0))


@eeglab_test(OPENBDF_SUITE, "test_pass_invalid_phys_min_max")
def test_upstream_openbdf_original_invalid_physical_limits(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_invalid_phys_min_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["PhysMin"], data["Head"]["DigMin"])
    np.testing.assert_array_equal(data["Head"]["PhysMax"], data["Head"]["DigMax"])


@eeglab_test(OPENBDF_SUITE, "test_pass_phys_min_larger_max")
def test_upstream_openbdf_original_reversed_physical_limits(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_sigprocfunc/openbdf/test_phys_min_larger_max.bdf"
    data, _ = eeglab_backend("openbdf", str(path), nargout=2)
    np.testing.assert_array_equal(data["Head"]["PhysMin"], data["Head"]["DigMin"])
    np.testing.assert_array_equal(data["Head"]["PhysMax"], data["Head"]["DigMax"])


@eeglab_test(READBDF_SUITE, "test_pass_general")
def test_upstream_readbdf_original_record_selection(eeglab_backend, eeglab_suite_root):
    path = eeglab_suite_root / "unittesting_binary/readbdf/test.bdf"
    bdf = eeglab_backend("openbdf", str(path))
    records = np.array([[1.0, 2.0, 4.0]])
    data, signal = eeglab_backend("readbdf", bdf, records, nargout=2)
    assert data["Record"].shape == (17, 256 * 3)
    assert data["Valid"].shape == (1, 256 * 3)
    np.testing.assert_array_equal(data["Idx"], records)
    assert signal.shape == (256, 17)

"""BrainVision loader tests, including the current EEGLAB wrapper port."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from eegprep import pop_loadbv
from eegprep.functions.popfunc.pop_fileio import pop_fileio
from tests.eeglab_tests import eeglab_test


_UPSTREAM_WRAPPER = "unittesting_binary/pop_loadbv/binary_pop_loadbv_wrapperTest.m"
_DTYPES = {
    "INT_16": np.dtype("<i2"),
    "UINT_16": np.dtype("<u2"),
    "IEEE_FLOAT_32": np.dtype("<f4"),
}


def _write_binary_brainvision(
    directory: Path,
    stem: str,
    data: np.ndarray,
    *,
    orientation: str = "MULTIPLEXED",
    binary_format: str = "INT_16",
    data_suffix: str = ".dat",
    labels: list[str] | None = None,
    references: list[str] | None = None,
    resolutions: list[float] | None = None,
    units: list[str] | None = None,
    coordinates: list[tuple[float, float, float]] | None = None,
    markers: list[str] | None = None,
    sampling_interval: float = 4000,
    declared_points: int | None = None,
    include_data_points: bool = True,
    segmentation_type: str | None = None,
    big_endian: bool = False,
) -> Path:
    channel_data = np.asarray(data)
    if channel_data.ndim != 2:
        raise ValueError("fixture data must be channel by sample")
    channel_count, point_count = channel_data.shape
    labels = labels or [f"Ch{index}" for index in range(1, channel_count + 1)]
    references = references or [""] * channel_count
    resolutions = resolutions or [1.0] * channel_count
    units = units or ["µV"] * channel_count
    data_name = f"{stem}{data_suffix}"
    marker_name = f"{stem}.vmrk"
    dtype = _DTYPES[binary_format].newbyteorder(">" if big_endian else "<")
    on_disk = channel_data.T if orientation == "MULTIPLEXED" else channel_data
    np.asarray(on_disk, dtype=dtype).tofile(directory / data_name)

    channel_lines = [
        f"Ch{index}={label},{reference},{resolution:.12g},{unit}"
        for index, (label, reference, resolution, unit) in enumerate(
            zip(labels, references, resolutions, units), start=1
        )
    ]
    coordinate_section = ""
    if coordinates is not None:
        coordinate_lines = [
            f"Ch{index}={radius:.12g},{theta:.12g},{phi:.12g}"
            for index, (radius, theta, phi) in enumerate(coordinates, start=1)
        ]
        coordinate_section = "\n[Coordinates]\n" + "\n".join(coordinate_lines) + "\n"
    data_points = point_count if declared_points is None else declared_points
    data_points_line = f"DataPoints={data_points}\n" if include_data_points else ""
    segmentation_line = f"SegmentationType={segmentation_type}\n" if segmentation_type else ""
    header = (
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "\n[Common Infos]\n"
        f"DataFile={data_name}\n"
        f"MarkerFile={marker_name}\n"
        "DataFormat=BINARY\n"
        "DataType=TIMEDOMAIN\n"
        f"DataOrientation={orientation}\n"
        f"NumberOfChannels={channel_count}\n"
        f"{data_points_line}"
        f"SamplingInterval={sampling_interval:.12g}\n"
        f"{segmentation_line}"
        "\n[Binary Infos]\n"
        f"BinaryFormat={binary_format}\n"
        f"UseBigEndianOrder={'YES' if big_endian else 'NO'}\n"
        "\n[Channel Infos]\n" + "\n".join(channel_lines) + "\n" + coordinate_section
    )
    header_path = directory / f"{stem}.vhdr"
    header_path.write_text(header, encoding="utf-8")
    _write_markers(directory / marker_name, data_name, markers or [])
    return header_path


def _write_ascii_brainvision(
    directory: Path,
    stem: str,
    data: np.ndarray,
    *,
    orientation: str,
    skip_columns: int = 0,
) -> Path:
    channel_data = np.asarray(data, dtype=float)
    channel_count, point_count = channel_data.shape
    rows = channel_data.T if orientation == "MULTIPLEXED" else channel_data
    lines = ["header row"]
    for index, row in enumerate(rows, start=1):
        values = " ".join(f"{value:.12g}" for value in row)
        lines.append(f"{index} {values}" if skip_columns else values)
    data_name = f"{stem}.dat"
    (directory / data_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    marker_name = f"{stem}.vmrk"
    header_path = directory / f"{stem}.vhdr"
    header_path.write_text(
        "Brain Vision Data Exchange Header File Version 1.0\n"
        "\n[Common Infos]\n"
        f"DataFile={data_name}\n"
        f"MarkerFile={marker_name}\n"
        "DataFormat=ASCII\n"
        "DataType=TIMEDOMAIN\n"
        f"DataOrientation={orientation}\n"
        f"NumberOfChannels={channel_count}\n"
        f"DataPoints={point_count}\n"
        "SamplingInterval=2000\n"
        "\n[ASCII Infos]\n"
        "SkipLines=1\n"
        f"SkipColumns={skip_columns}\n"
        "DecimalSymbol=.\n"
        "\n[Channel Infos]\n"
        + "\n".join(f"Ch{index}=E{index},,0.5,µV" for index in range(1, channel_count + 1))
        + "\n",
        encoding="utf-8",
    )
    _write_markers(directory / marker_name, data_name, [])
    return header_path


def _write_markers(path: Path, data_name: str, markers: list[str]) -> None:
    marker_lines = "\n".join(f"Mk{index}={marker}" for index, marker in enumerate(markers, start=1))
    path.write_text(
        "Brain Vision Data Exchange Marker File, Version 1.0\n"
        "\n[Common Infos]\n"
        f"DataFile={data_name}\n"
        "\n[Marker Infos]\n"
        f"{marker_lines}\n",
        encoding="utf-8",
    )


def _assert_continuous_eeg(eeg: dict[str, Any], expected: np.ndarray, srate: float) -> None:
    np.testing.assert_allclose(eeg["data"], expected, rtol=1e-6, atol=1e-8)
    assert eeg["data"].shape == expected.shape
    assert eeg["nbchan"] == expected.shape[0]
    assert eeg["pnts"] == expected.shape[1]
    assert eeg["trials"] == 1
    assert eeg["srate"] == pytest.approx(srate)
    assert eeg["xmin"] == 0
    assert eeg["xmax"] == pytest.approx((expected.shape[1] - 1) / srate)
    np.testing.assert_allclose(eeg["times"], np.arange(expected.shape[1]) / srate * 1000)


@eeglab_test(_UPSTREAM_WRAPPER, "test_test_pop_loadbv")
def test_pop_loadbv_ports_all_eight_active_upstream_load_calls(tmp_path: Path) -> None:
    selected_raw = np.arange(32 * 5, dtype=np.int16).reshape(32, 5) - 50
    selected_header = _write_binary_brainvision(
        tmp_path,
        "brainvision_genericdataformat_binarymultiplexed_int16",
        selected_raw,
    )
    selected = pop_loadbv(tmp_path, selected_header.name, 1, list(range(1, 33)))
    _assert_continuous_eeg(selected, selected_raw, 250.0)
    assert "binarymultiplexed_int16.vhdr', 1, [1 2 3" in selected["history"]

    located_raw = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    located_header = _write_binary_brainvision(
        tmp_path,
        "BVA_withchanlocs",
        located_raw,
        labels=["Cz", "F4"],
        coordinates=[(1, 0, 0), (1, 60, 51)],
    )
    located = pop_loadbv(tmp_path, located_header.name)
    _assert_continuous_eeg(located, located_raw, 250.0)
    assert located["chanlocs"][0]["Z"] == pytest.approx(1)
    assert located["chanlocs"][1]["X"] == pytest.approx(np.sin(np.deg2rad(60)) * np.sin(np.deg2rad(51)))
    assert located["chanlocs"][1]["Y"] == pytest.approx(-np.sin(np.deg2rad(60)) * np.cos(np.deg2rad(51)))
    assert located["chanlocs"][1]["Z"] == pytest.approx(0.5)

    cases = [
        ("brainvision_genericdataformat_binarymultiplexed_ieee32", "MULTIPLEXED", "IEEE_FLOAT_32", ".dat"),
        ("brainvision_genericdataformat_binarymultiplexed_int16", "MULTIPLEXED", "INT_16", ".dat"),
        ("brainvision_genericdataformat_binaryvectorized_ieee", "VECTORIZED", "IEEE_FLOAT_32", ".dat"),
        ("brainvision_genericdataformat_binaryvectorized_int16", "VECTORIZED", "INT_16", ".dat"),
        ("brainvision_recorder_acquisitiondataformat", "MULTIPLEXED", "INT_16", ".eeg"),
    ]
    for index, (stem, orientation, binary_format, data_suffix) in enumerate(cases, start=1):
        if binary_format == "IEEE_FLOAT_32":
            raw = np.array([[0.25, -1.5, 3.125], [4.5, 2.25, -0.75]], dtype=np.float32) * index
        else:
            raw = np.array([[1, -2, 30], [40, 5, -6]], dtype=np.int16) * index
        header = _write_binary_brainvision(
            tmp_path,
            stem,
            raw,
            orientation=orientation,
            binary_format=binary_format,
            data_suffix=data_suffix,
            resolutions=[0.5, 2.0],
            include_data_points=stem != "brainvision_recorder_acquisitiondataformat",
        )
        if stem == "brainvision_recorder_acquisitiondataformat":
            header.write_text(
                header.read_text(encoding="utf-8")
                + "\n[Comment]\nRecorder free-form metadata\n===========================\n"
                + "Number of channels: 2\n",
                encoding="utf-8",
            )
        eeg = pop_loadbv(tmp_path, header.name)
        _assert_continuous_eeg(eeg, raw * np.array([[0.5], [2.0]]), 250.0)
        assert eeg["etc"]["brainvision"]["data_orientation"] == orientation
        assert eeg["comments"] == f"Original file: {stem}{data_suffix}"

    export_raw = np.array([[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]], dtype=np.float32)
    export_header = _write_binary_brainvision(
        tmp_path,
        "EEGLAB_export",
        export_raw,
        orientation="VECTORIZED",
        binary_format="IEEE_FLOAT_32",
        markers=[
            "New Segment,,1,0,0,0",
            "Time 0,Stimulus,2.25,0,0,0",
            "New Segment,,4,0,0,0",
            "Stimulus,square,5.5,0,0,0",
        ],
        segmentation_type="MARKERBASED",
    )
    export_marker = tmp_path / "EEGLAB_export.vmrk"
    marker_text = export_marker.read_text(encoding="utf-8")
    for marker_number in range(2, 5):
        marker_text = marker_text.replace(f"Mk{marker_number}=", "Mk1=")
    export_marker.write_text(marker_text, encoding="utf-8")
    exported = pop_loadbv(tmp_path, export_header.name)
    expected_export = export_raw.reshape(2, 2, 3).transpose(0, 2, 1)
    np.testing.assert_array_equal(exported["data"], expected_export)
    assert exported["data"].shape == (2, 3, 2)
    assert exported["pnts"] == 3
    assert exported["trials"] == 2
    assert exported["xmin"] == pytest.approx(-1.25 / 250)
    assert exported["xmax"] == pytest.approx(0.75 / 250)
    assert [(event["type"], event["latency"], event["epoch"]) for event in exported["event"]] == [
        ("TLE", 2.25, 1),
        ("square", 5.5, 2),
    ]


def test_pop_loadbv_normalizes_units_selects_samples_and_preserves_markers(tmp_path: Path) -> None:
    raw = np.array(
        [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300, 400, 500],
            [1000, 2000, 3000, 4000, 5000],
        ],
        dtype=np.int16,
    )
    header = _write_binary_brainvision(
        tmp_path,
        "units",
        raw,
        labels=[r"Fp\1Left", "B", "C", "D"],
        references=["Cz", "", "", ""],
        resolutions=[1e-6, 1e-3, 2, 1],
        units=["V", "mV", "uV", "nV"],
        markers=[
            "New Segment,,1,1,0,20260913123000123456",
            "Stimulus,S  1,2,3,4",
            r"Response,correct\1fast,4,2,0,,hidden",
            "Stimulus,outside,5,1,0",
        ],
    )

    eeg, command = pop_loadbv(tmp_path, header.name, [2, 4], [4, 2], return_com=True)

    expected = np.vstack([raw[3, 1:4] * 0.001, raw[1, 1:4]])
    _assert_continuous_eeg(eeg, expected, 250.0)
    assert [loc["labels"] for loc in eeg["chanlocs"]] == ["D", "B"]
    assert [loc["bvunit"] for loc in eeg["chanlocs"]] == ["nV", "mV"]
    assert [loc["unit"] for loc in eeg["chanlocs"]] == ["µV", "µV"]
    assert [loc["urchan"] for loc in eeg["chanlocs"]] == [3, 1]
    assert [event["type"] for event in eeg["event"]] == ["S  1", "correct,fast"]
    assert [event["latency"] for event in eeg["event"]] == [1, 3]
    assert [event["duration"] for event in eeg["event"]] == [3, 2]
    assert eeg["event"][0]["channel"] == 1
    assert eeg["event"][0]["bvchannel"] == 4
    assert [event["bvmknum"] for event in eeg["event"]] == [2, 3]
    assert [event["urevent"] for event in eeg["event"]] == [0, 1]
    assert eeg["event"][1]["visible"] == "hidden"
    assert [event["type"] for event in eeg["urevent"]] == ["S  1", "correct,fast"]
    assert command == f"EEG = pop_loadbv('{tmp_path.as_posix()}', 'units.vhdr', [2 4], [4 2]);"
    assert eeg["history"] == command

    complete = pop_loadbv(header)
    np.testing.assert_allclose(
        complete["data"],
        raw * np.array([[1.0], [1.0], [2.0], [0.001]]),
    )
    assert complete["chanlocs"][0]["labels"] == "Fp,Left"
    assert complete["chanlocs"][0]["ref"] == "Cz"
    assert complete["event"][0]["type"] == "boundary"
    assert np.isnan(complete["event"][0]["duration"])
    assert complete["event"][0]["bvtime"] == "20260913123000123456"

    tail = pop_loadbv(header, srange=3)
    np.testing.assert_allclose(
        tail["data"],
        raw[:, 2:] * np.array([[1.0], [1.0], [2.0], [0.001]]),
    )
    assert tail["pnts"] == 3
    assert [(event["type"], event["latency"]) for event in tail["event"]] == [
        ("correct,fast", 2),
        ("outside", 3),
    ]


def test_pop_loadbv_metadata_only_keeps_dimensions_channels_and_events(tmp_path: Path) -> None:
    header = _write_binary_brainvision(
        tmp_path,
        "metadata",
        np.arange(18, dtype=np.int16).reshape(3, 6),
        markers=["Stimulus,A,3,1,0"],
    )

    eeg, command = pop_loadbv(header, srange=[2, 5], chans=[3, 1], metadata=True, return_com=True)

    assert eeg["data"].shape == (0,)
    assert eeg["nbchan"] == 2
    assert eeg["pnts"] == 4
    assert eeg["trials"] == 1
    assert eeg["srate"] == 250
    assert [loc["labels"] for loc in eeg["chanlocs"]] == ["Ch3", "Ch1"]
    assert [(event["type"], event["latency"]) for event in eeg["event"]] == [("A", 2)]
    assert eeg["etc"]["brainvision"]["metadata_only"] is True
    assert command.endswith("'metadata.vhdr', [2 5], [3 1], true);")


@pytest.mark.parametrize("orientation", ["MULTIPLEXED", "VECTORIZED"])
def test_pop_loadbv_reads_ascii_orientations_with_header_rows(
    tmp_path: Path,
    orientation: str,
) -> None:
    raw = np.array([[1.25, -2.5, 3.75], [4.0, 5.5, -6.25]])
    header = _write_ascii_brainvision(tmp_path, orientation.lower(), raw, orientation=orientation, skip_columns=1)

    eeg = pop_loadbv(header)

    _assert_continuous_eeg(eeg, raw * 0.5, 500.0)


def test_pop_loadbv_honors_big_endian_binary_headers(tmp_path: Path) -> None:
    raw = np.array([[1, 256, -2], [1024, -1024, 7]], dtype=np.int16)
    header = _write_binary_brainvision(tmp_path, "bigendian", raw, big_endian=True)

    eeg = pop_loadbv(header)

    _assert_continuous_eeg(eeg, raw, 250.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("DataOrientation=MULTIPLEXED", "DataOrientation=UNKNOWN"), "orientation"),
        (lambda text: text.replace("DataType=TIMEDOMAIN", "DataType=FREQUENCYDOMAIN"), "data type"),
        (lambda text: text.replace("SamplingInterval=4000", "SamplingInterval=0"), "SamplingInterval"),
        (lambda text: text.replace("BinaryFormat=INT_16", "BinaryFormat=INT_32"), "binary format"),
        (lambda text: text.replace("Ch2=Ch2,,1,µV\n", ""), "missing Ch2"),
        (lambda text: text.replace("DataPoints=3", "DataPoints=4"), "declares 4 samples"),
    ],
)
def test_pop_loadbv_rejects_malformed_headers(
    tmp_path: Path,
    change: Callable[[str], str],
    message: str,
) -> None:
    header = _write_binary_brainvision(tmp_path, "malformed", np.ones((2, 3), dtype=np.int16))
    header.write_text(change(header.read_text(encoding="utf-8")), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        pop_loadbv(header)


def test_pop_loadbv_rejects_truncated_binary_and_invalid_selections(tmp_path: Path) -> None:
    header = _write_binary_brainvision(tmp_path, "truncated", np.ones((2, 3), dtype=np.int16))
    data_path = tmp_path / "truncated.dat"
    data_path.write_bytes(data_path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="truncated"):
        pop_loadbv(header)

    valid = _write_binary_brainvision(tmp_path, "valid", np.ones((2, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="srange"):
        pop_loadbv(valid, srange=[0, 2])
    with pytest.raises(ValueError, match="srange"):
        pop_loadbv(valid, srange=[2, 4])
    with pytest.raises(ValueError, match="chans"):
        pop_loadbv(valid, chans=[3])
    with pytest.raises(ValueError, match="integer"):
        pop_loadbv(valid, chans=[1.5])


def test_pop_loadbv_warns_and_loads_when_marker_file_is_missing(tmp_path: Path) -> None:
    header = _write_binary_brainvision(tmp_path, "nomarkers", np.ones((1, 3), dtype=np.int16))
    (tmp_path / "nomarkers.vmrk").unlink()

    with pytest.warns(RuntimeWarning, match="marker file not found"):
        eeg = pop_loadbv(header)

    assert eeg["event"].size == 0
    assert eeg["urevent"].size == 0


def test_pop_loadbv_accepts_data_path_and_resolves_companions_case_insensitively(tmp_path: Path) -> None:
    header = _write_binary_brainvision(
        tmp_path,
        "case",
        np.array([[1, 2, 3]], dtype=np.int16),
        data_suffix=".eeg",
    )
    header.rename(tmp_path / "CASE.VHDR")
    (tmp_path / "case.eeg").rename(tmp_path / "CASE.EEG")
    (tmp_path / "case.vmrk").rename(tmp_path / "CASE.VMRK")

    eeg = pop_loadbv(tmp_path / "case.eeg")

    _assert_continuous_eeg(eeg, np.array([[1, 2, 3]]), 250.0)


def test_pop_fileio_routes_brainvision_headers_through_the_standalone_loader(tmp_path: Path) -> None:
    raw = np.array([[2, 4, 6], [1, 3, 5]], dtype=np.int16)
    header = _write_binary_brainvision(
        tmp_path,
        "fileio",
        raw,
        orientation="VECTORIZED",
        resolutions=[2, 0.5],
    )

    eeg, command = pop_fileio(header, channels=[2], samples=[2, 3], return_com=True)

    _assert_continuous_eeg(eeg, np.array([[1.5, 2.5]]), 250.0)
    assert command == f"EEG = pop_fileio('{header.as_posix()}', 'channels', [2], 'samples', [2 3]);"
    assert eeg["history"] == command

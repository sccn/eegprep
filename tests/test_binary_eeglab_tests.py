"""Behavioral ports of applicable tests from EEGLAB's current binary suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib
import pytest
from pyedflib import highlevel

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc.pop_biosig import pop_biosig
from eegprep.functions.popfunc.pop_chancoresp import pop_chancoresp
from eegprep.functions.popfunc.pop_chanevent import pop_chanevent
from eegprep.functions.popfunc.pop_importpres import pop_importpres
from eegprep.functions.popfunc.pop_snapread import pop_snapread
from eegprep.functions.sigprocfunc.snapread import snapread
from tests.eeglab_tests import eeglab_test


BINARY_SUITE = "unittesting_binary"


def _write_edf_family(path: Path, data: np.ndarray, srate: int, *, bdf: bool = False) -> None:
    headers = highlevel.make_signal_headers(
        [f"E{index}" for index in range(1, data.shape[0] + 1)],
        sample_frequency=srate,
        physical_min=-200,
        physical_max=200,
    )
    file_type = pyedflib.FILETYPE_BDFPLUS if bdf else pyedflib.FILETYPE_EDFPLUS
    assert highlevel.write_edf(str(path), data, headers, file_type=file_type)


@eeglab_test(f"{BINARY_SUITE}/pop_biosig/binary_pop_biosig_wrapperTest.m", "test_test_pop_biosig")
def test_pop_biosig_reads_bdf_blockrange_with_exact_metadata(tmp_path: Path) -> None:
    srate = 8
    data = np.vstack(
        [
            np.linspace(-100, 100, srate * 12),
            np.sin(np.arange(srate * 12) / 4) * 50,
        ]
    )
    filename = tmp_path / "recording.bdf"
    _write_edf_family(filename, data, srate, bdf=True)

    complete = pop_biosig(filename)
    cropped, command = pop_biosig(filename, blockrange=[1, 10], return_com=True)

    assert cropped["data"].shape == (2, 9 * srate)
    assert cropped["nbchan"] == 2
    assert cropped["pnts"] == 9 * srate
    assert cropped["trials"] == 1
    assert cropped["srate"] == srate
    assert cropped["xmin"] == 0
    assert cropped["xmax"] == pytest.approx((9 * srate - 1) / srate)
    assert [loc["labels"] for loc in cropped["chanlocs"]] == ["E1", "E2"]
    np.testing.assert_array_equal(cropped["data"], complete["data"][:, srate : 10 * srate])
    assert "'blockrange', [1 10]" in command
    assert cropped["history"] == command


@eeglab_test(
    f"{BINARY_SUITE}/pop_biosig/binary_pop_biosig_wrapperTest.m",
    "test_test_pop_biosig_timerange",
)
def test_pop_biosig_adjacent_edf_timeranges_are_sample_continuous(tmp_path: Path) -> None:
    srate = 256
    samples = np.arange(61 * srate, dtype=float)
    data = (np.sin(samples / 37) * 100)[np.newaxis, :]
    filename = tmp_path / "recording.edf"
    _write_edf_family(filename, data, srate)

    first = pop_biosig(filename, blockrange=[0, 30])
    second = pop_biosig(filename, blockrange=[29, 60])

    assert first["pnts"] == 30 * srate
    assert second["pnts"] == 31 * srate
    np.testing.assert_array_equal(first["data"][0, -srate:], second["data"][0, :srate])


CHANCORRESP_SOURCE = f"{BINARY_SUITE}/pop_chancoresp/binary_pop_chancoresp_wrapperTest.m"


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_autoselect_fiducials")
def test_pop_chancoresp_autoselects_fiducials_case_insensitively() -> None:
    left, right = pop_chancoresp(
        ["Nz", "lpa", "rpa", "x"],
        ["nZ", "rpa", "lpa", "x"],
        "gui",
        "off",
        "autoselect",
        "fiducials",
    )

    assert left == [1, 2, 3]
    assert right == [1, 3, 2]


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_autoselect_none")
def test_pop_chancoresp_autoselect_none_returns_no_pairs() -> None:
    assert pop_chancoresp(["a", "b", "c"], ["a", "x", "b"], "gui", "off", "autoselect", "none") == (
        [],
        [],
    )


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_chanlists_not_empty")
def test_pop_chancoresp_preserves_explicit_pairs() -> None:
    result = pop_chancoresp(
        ["a", "b"],
        ["x", "y"],
        "gui",
        "off",
        "chanlist1",
        [1, 2],
        "chanlist2",
        [2, 1],
    )

    assert result == ([1, 2], [2, 1])


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_clear")
def test_pop_chancoresp_clear_returns_unpaired_display_rows() -> None:
    left, right = pop_chancoresp("clear", ["a", "b", "c"], ["a", "b", "x"])

    assert left == [" 1 -   a", " 2 -   b", " 3 -   c"]
    assert right == [" 1 -   a", " 2 -   b", " 3 -   x"]


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_invalid_fiducials")
def test_pop_chancoresp_invalid_fiducials_return_no_pairs() -> None:
    result = pop_chancoresp(["x"], ["x"], "gui", "off", "autoselect", "fiducials")

    assert result == ([], [])


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_labels_only")
def test_pop_chancoresp_pairs_matching_labels_by_default() -> None:
    result = pop_chancoresp(["a", "b", "c"], ["a", "x", "b"], "gui", "off")

    assert result == ([1, 2], [1, 3])


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_pair")
def test_pop_chancoresp_pair_updates_text_and_correspondences() -> None:
    result = pop_chancoresp("pair", 2, 3, ["a", "b", "c"], ["a", "b", "x"], [], [], "", "")

    assert result == (" 2 -   b   ->  3 -   x", " 3 -   x   ->  2 -   b", [2], [3])


@eeglab_test(CHANCORRESP_SOURCE, "test_pass_unpair")
def test_pop_chancoresp_unpair_removes_correspondence_and_updates_text() -> None:
    result = pop_chancoresp(
        "unpair",
        2,
        3,
        ["a", "b", "c"],
        ["a", "b", "x"],
        [1, 2, 3],
        [1, 3, 2],
        "",
        "",
    )

    assert result == (" 2 -   b", " 3 -   x", [1, 3], [1, 2])


@eeglab_test(CHANCORRESP_SOURCE, "test_test_pop_chancoresp")
def test_pop_chancoresp_covers_the_upstream_option_matrix() -> None:
    first = ["Nz", "LPA", "RPA", *[f"E{index}" for index in range(4, 33)]]
    same = list(first)
    different = ["Nz", "LPA", "RPA", *[f"X{index}" for index in range(4, 33)]]
    longer = [*first, *[f"E{index}" for index in range(33, 69)]]
    explicit_pairs = [
        (list(range(1, 17)), list(range(1, 17))),
        (list(range(1, 17)), list(range(17, 33))),
        ([1, 2, 3, 4, 5], [17, 6, 1, 30, 5]),
    ]

    for second, all_pairs in [
        (same, (list(range(1, 33)), list(range(1, 33)))),
        (different, ([1, 2, 3], [1, 2, 3])),
        (longer, (list(range(1, 33)), list(range(1, 33)))),
    ]:
        assert pop_chancoresp(first, second, "gui", "off") == all_pairs
        assert pop_chancoresp(first, second, "gui", "off", "autoselect", "none") == ([], [])
        assert pop_chancoresp(first, second, "gui", "off", "autoselect", "all") == all_pairs
        assert pop_chancoresp(first, second, "gui", "off", "autoselect", "fiducials") == (
            [1, 2, 3],
            [1, 2, 3],
        )
        for mode in ["none", "all", "fiducials"]:
            for left, right in explicit_pairs:
                result = pop_chancoresp(
                    first,
                    second,
                    "gui",
                    "off",
                    "autoselect",
                    mode,
                    "chanlist1",
                    left,
                    "chanlist2",
                    right,
                )
                assert result == (left, right)


@eeglab_test(f"{BINARY_SUITE}/pop_chanevent/binary_pop_chanevent_wrapperTest.m", "test_test_pop_chanevent")
def test_pop_chanevent_covers_the_upstream_33_case_option_matrix() -> None:
    trigger = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0], dtype=float)
    eeg = eeg_from_data(
        np.vstack([np.arange(trigger.size), trigger]),
        srate=100,
        chanlocs=[{"labels": "Cz"}, {"labels": "TRIG"}],
    )
    edge_cases = [
        ("both", 1, False, [2, 5, 6, 9], None),
        ("leading", 1, False, [2, 6], None),
        ("trailing", 1, False, [5, 9], None),
        ("both", 10, False, [2, 9], None),
        ("leading", 10, False, [2], None),
        ("trailing", 10, False, [9], None),
        ("leading", 1, True, [2, 6], [2, 2]),
        ("leading", 10, True, [2], [6]),
    ]

    for delchan in ["on", "off"]:
        for delevent in ["on", "off"]:
            for edge, edgelen, duration, latencies, durations in edge_cases:
                out = pop_chanevent(
                    eeg,
                    2,
                    "edge",
                    edge,
                    "edgelen",
                    edgelen,
                    "oper",
                    "",
                    "duration",
                    "on" if duration else "off",
                    "delchan",
                    delchan,
                    "delevent",
                    delevent,
                    "nbtype",
                    np.nan,
                    "typename",
                    "TRIG",
                )
                events = [dict(event) for event in out["event"]]
                assert [event["latency"] for event in events] == latencies
                assert [event["type"] for event in events] == ["TRIG"] * len(latencies)
                assert [event["urevent"] for event in events] == list(range(len(events)))
                assert out["nbchan"] == (1 if delchan == "on" else 2)
                if durations is not None:
                    assert [event["duration"] for event in events] == durations

    default = pop_chanevent(eeg, 2)
    assert [event["latency"] for event in default["event"]] == [2, 5, 6, 9]
    assert [event["type"] for event in default["event"]] == ["chan2"] * 4
    assert default["data"].shape == (1, trigger.size)


@eeglab_test(f"{BINARY_SUITE}/pop_importpres/binary_pop_importpres_wrapperTest.m", "test_test_pop_importpres")
def test_pop_importpres_covers_the_upstream_five_call_forms(tmp_path: Path) -> None:
    filename = tmp_path / "experiment.LOG"
    filename.write_text(
        "Scenario demo\n"
        "Logfile demo\n"
        "Subject demo\n"
        "Trial demo\n"
        "Timestamp demo\n"
        "Subject\tEvent Type\tCode\tTime\tDuration\n"
        "S01\tPicture\t11\t2\t1\n"
        "S01\tResponse\t12\t3\t2\n",
        encoding="utf-8",
    )
    eeg = eeg_from_data(np.zeros((1, 405)), srate=100)

    default = pop_importpres(eeg, filename)
    labels_only = pop_importpres(eeg, filename, "Event Type", "Time", "None")
    with_duration = pop_importpres(eeg, filename, "Event Type", "Time", "Duration")
    explicit = pop_importpres(
        eeg,
        filename,
        "Event Type",
        "Time",
        "None",
        0,
        "timeunit",
        1,
        "append",
        "no",
        "indices",
        [],
        "align",
        0,
        "optimalign",
        "on",
    )
    skipped = pop_importpres(
        eeg,
        filename,
        "Event Type",
        "Time",
        "None",
        0,
        "skipline",
        5,
        "timeunit",
        1,
        "append",
        "no",
        "indices",
        [],
        "align",
        0,
        "optimalign",
        "on",
    )

    assert [event["type"] for event in default["event"]] == [11, 12]
    assert [event["latency"] for event in default["event"]] == pytest.approx([1.02, 1.03])
    assert [event["type"] for event in labels_only["event"]] == ["Picture", "Response"]
    assert all("duration" not in event for event in labels_only["event"])
    assert [event["duration"] for event in with_duration["event"]] == pytest.approx([0.01, 0.02])
    assert [event["latency"] for event in explicit["event"]] == [201, 301]
    assert [(event["type"], event["latency"]) for event in skipped["event"]] == [
        (event["type"], event["latency"]) for event in explicit["event"]
    ]


def _write_snapmaster(path: Path) -> np.ndarray:
    nframes = 405
    values = np.vstack(
        [
            np.r_[np.zeros(2), np.ones(2) * 3, np.zeros(396), np.ones(2) * 3, np.zeros(3)],
            np.arange(nframes, dtype=float),
            -np.arange(nframes, dtype=float),
        ]
    ).astype("<f4")
    header = b'"NCHAN%"=3\n"NUM.POINTS"=405\n"ACT.FREQ"=100\n"TR"\n2026-06-05\n'
    path.write_bytes(header + b"\xaa" + values.tobytes(order="F"))
    return values[1:]


@eeglab_test(f"{BINARY_SUITE}/snapread/binary_snapread_wrapperTest.m", "test_test_snapread")
def test_snapread_reads_default_and_seeked_binary_frames(tmp_path: Path) -> None:
    filename = tmp_path / "TEST.SMA"
    expected = _write_snapmaster(filename)

    complete, params, events, header = snapread(filename)
    after_400, seeked_params, seeked_events, _ = snapread(filename, 400)
    after_one, _, _, _ = snapread(filename, 1)

    np.testing.assert_array_equal(complete, expected)
    np.testing.assert_array_equal(after_400, expected[:, 400:])
    np.testing.assert_array_equal(after_one, expected[:, 1:])
    assert params.tolist() == [2, 405, 100]
    assert seeked_params.tolist() == [2, 5, 100]
    assert np.flatnonzero(events).tolist() == [2, 400]
    assert np.flatnonzero(seeked_events).tolist() == []
    assert '"NCHAN%"=3' in header


@eeglab_test(f"{BINARY_SUITE}/pop_snapread/binary_pop_snapread_wrapperTest.m", "test_test_pop_snapread")
def test_pop_snapread_applies_each_upstream_gain_and_builds_eeg_metadata(tmp_path: Path) -> None:
    filename = tmp_path / "TEST.SMA"
    expected = _write_snapmaster(filename)

    default = pop_snapread(filename)
    gain_400 = pop_snapread(filename, 400)
    gain_one, command = pop_snapread(filename, 1, return_com=True)

    np.testing.assert_array_equal(default["data"], expected)
    np.testing.assert_array_equal(gain_400["data"], expected * 400)
    np.testing.assert_array_equal(gain_one["data"], expected)
    assert gain_one["nbchan"] == 2
    assert gain_one["pnts"] == 405
    assert gain_one["trials"] == 1
    assert gain_one["srate"] == 100
    assert [event["latency"] for event in gain_one["event"]] == [3, 401]
    assert [event["type"] for event in gain_one["event"]] == [1, 1]
    assert gain_one["history"] == command

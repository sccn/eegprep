"""Current eeglab_tests coverage for importing epoch metadata."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest

from eegprep.functions.popfunc._file_io import eeg_from_data
from eegprep.functions.popfunc.pop_importepoch import pop_importepoch
from tests.eeglab_tests import eeglab_test


UPSTREAM = "unittesting_popfunc/pop_importepoch/popfunc_pop_importepoch_wrapperTest.m"


def _epoched_eeg() -> dict:
    eeg = eeg_from_data(np.zeros((1, 100, 3), dtype=float), srate=100, xmin=-0.2)
    eeg["event"] = [{"type": "old", "latency": 10.0, "duration": 0, "epoch": 1, "urevent": 0}]
    eeg["urevent"] = [{"type": "old", "latency": 10.0, "duration": 0, "epoch": 1}]
    return eeg


def _write_epoch_table(path: Path) -> Path:
    path.write_text(
        "Epoch\tResponse\tResponse_latency\n1\tCorrect\t250\n2\tWrong\t500\n3\tCorrect\t750\n",
        encoding="utf-8",
    )
    return path


_EPOCH_ROWS = [[1, "Correct", 250], [2, "Wrong", 500], [3, "Correct", 750]]
_EPOCH_ROWS_WITH_DURATION = [
    [1, "Correct", 250, 100],
    [2, "Wrong", 500, 200],
    [3, "Correct", 750, 300],
]


@eeglab_test(UPSTREAM, "test_test_pop_importepoch")
def test_reference_importepoch_all_four_original_cases(eeglab_backend, eeglab_suite_root, eeglab_working_directory):
    shutil.copyfile(
        eeglab_suite_root / "unittesting_popfunc/pop_importepoch/epochinfo.txt",
        eeglab_working_directory / "epochinfo.txt",
    )
    chanlocs = eeglab_backend("readlocs", str(eeglab_suite_root / "eeglab/sample_data/eeglab_chan32.locs"))
    rng = np.random.default_rng()
    eeg = eeglab_backend(
        "pop_importdata",
        "setname",
        "UnitTesting",
        "data",
        rng.random((32, 256, 10)),
        "dataformat",
        "array",
        "chanlocs",
        chanlocs,
        "nbchan",
        32.0,
        "pnts",
        256.0,
        "srate",
        256.0,
        "icaweights",
        rng.random((32, 32)),
        "icasphere",
        rng.random((32, 32)),
        "comments",
        "Test Passes!",
    )
    eeg["trials"] = 10.0
    fields = np.array([["epoch", "response", "rt"]], dtype=object)
    latencyfields = np.array([["rt"]], dtype=object)
    eeglab_backend(
        "pop_importepoch",
        eeg,
        "epochinfo.txt",
        fields,
        "latencyfields",
        latencyfields,
        "timeunit",
        0.001,
        "headerlines",
        np.array([[1.0]]),
    )
    eeglab_backend(
        "pop_importepoch",
        eeg,
        "epochinfo.txt",
        fields,
        "typefield",
        "response",
        "timeunit",
        1e-3,
        "latencyfields",
        latencyfields,
        "headerlines",
        np.array([[1.0]]),
        "clearevents",
        "on",
    )
    rows = [
        [float(index), "Wrong" if index == 6 else "Correct", float(latency)]
        for index, latency in enumerate((502, 477, 453, 612, 430, 525, 498, 601, 398, 573), start=1)
    ]
    eeglab_backend(
        "pop_importepoch",
        eeg,
        np.array(rows, dtype=object),
        fields,
        "typefield",
        "response",
        "timeunit",
        1e-3,
        "latencyfields",
        latencyfields,
        "headerlines",
        np.array([[0.0]]),
        "clearevents",
        "on",
    )
    eeglab_backend(
        "pop_importepoch",
        eeg,
        np.array([row + [1.0] for row in rows], dtype=object),
        np.array([["epoch", "response", "rt", "dr"]], dtype=object),
        "typefield",
        "response",
        "durationfields",
        np.array([["dr"]], dtype=object),
        "timeunit",
        1e-3,
        "latencyfields",
        latencyfields,
        "headerlines",
        np.array([[0.0]]),
        "clearevents",
        "on",
    )


@pytest.mark.parametrize("case", [1, 2, 3, 4], ids=["case-1", "case-2", "case-3", "case-4"])
def test_current_pop_importepoch_option_cases(tmp_path: Path, case: int) -> None:
    epoch_file = _write_epoch_table(tmp_path / "epochinfo.txt")
    if case == 1:
        source = epoch_file
        fields = ["epoch", "response", "rt"]
        options = ("latencyfields", ["rt"], "timeunit", 1e-3, "headerlines", [1])
    elif case == 2:
        source = epoch_file
        fields = ["epoch", "response", "rt"]
        options = (
            "typefield",
            "response",
            "timeunit",
            1e-3,
            "latencyfields",
            ["rt"],
            "headerlines",
            [1],
            "clearevents",
            "on",
        )
    elif case == 3:
        source = _EPOCH_ROWS
        fields = ["epoch", "response", "rt"]
        options = (
            "typefield",
            "response",
            "timeunit",
            1e-3,
            "latencyfields",
            ["rt"],
            "headerlines",
            [0],
            "clearevents",
            "on",
        )
    else:
        source = _EPOCH_ROWS_WITH_DURATION
        fields = ["epoch", "response", "rt", "dr"]
        options = (
            "typefield",
            "response",
            "durationfields",
            ["dr"],
            "timeunit",
            1e-3,
            "latencyfields",
            ["rt"],
            "headerlines",
            [0],
            "clearevents",
            "on",
        )

    output = pop_importepoch(_epoched_eeg(), source, fields, *options)

    events = [dict(event) for event in output["event"]]
    assert len(events) == 6
    assert all(event["type"] != "old" for event in events)
    assert [event["latency"] for event in events] == pytest.approx([21, 46, 121, 171, 221, 296])
    expected_locking_types = ["TLE", "TLE", "TLE"] if case == 1 else ["Correct", "Wrong", "Correct"]
    assert [events[index]["type"] for index in (0, 2, 4)] == expected_locking_types
    assert [events[index]["type"] for index in (1, 3, 5)] == ["rt", "rt", "rt"]
    assert [output["epoch"][index]["response"] for index in range(3)] == ["Correct", "Wrong", "Correct"]
    assert [output["epoch"][index]["rt"] for index in range(3)] == [250, 500, 750]
    if case == 1:
        assert [event["response"] for event in events] == ["Correct", "Correct", "Wrong", "Wrong", "Correct", "Correct"]
    else:
        assert all("response" not in event for event in events)
    expected_durations = [0, 10, 0, 20, 0, 30] if case == 4 else [0, 0, 0, 0, 0, 0]
    assert [event["duration"] for event in events] == pytest.approx(expected_durations)
    assert all(output["urevent"][event["urevent"]]["latency"] == event["latency"] for event in events)

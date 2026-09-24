"""Ports of the current EEGLAB segmented EGI MATLAB import test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from eegprep import pop_importegimat
from tests.eeglab_tests import eeglab_test


UPSTREAM_WRAPPER = "unittesting_binary/pop_importegimat/binary_pop_importegimat_wrapperTest.m"


def _segmented_fixture(path: Path, *, trials: int = 3, pnts: int = 12) -> list[np.ndarray]:
    segments = []
    variables: dict[str, object] = {"samplingRate": np.array([[250.0]])}
    for trial in range(trials):
        segment = np.arange(129 * pnts, dtype=np.float64).reshape(129, pnts) + trial * 10_000
        segment[-1] = 0
        segments.append(segment)
        variables[f"CM1CP1POS_Segment{trial + 1}"] = segment
    variables["ECI"] = np.array([["ignored by EEGLAB"]], dtype=object)
    savemat(path, variables)
    return segments


def test_pop_importegimat_loads_segmented_netstation_export(tmp_path: Path) -> None:
    """Turn the upstream assertion-free smoke call into an observable contract."""
    path = tmp_path / "segmented_matlab.mat"
    segments = _segmented_fixture(path)

    eeg, command = pop_importegimat(path, return_com=True)

    expected = np.stack([segment[:-1] for segment in segments], axis=2).astype(np.float32)
    np.testing.assert_array_equal(eeg["data"], expected)
    assert eeg["data"].dtype == np.float32
    assert (eeg["nbchan"], eeg["pnts"], eeg["trials"]) == (128, 12, 3)
    assert eeg["srate"] == 250
    assert eeg["xmin"] == 0
    assert eeg["xmax"] == pytest.approx(11 / 250)
    np.testing.assert_allclose(eeg["times"], np.arange(12) / 250 * 1000)

    events = list(eeg["event"])
    assert [event["type"] for event in events] == ["CM1CP1POS"] * 3
    assert [event["latency"] for event in events] == [1, 13, 25]
    assert [event["epoch"] for event in events] == [1, 2, 3]
    assert [event["urevent"] for event in events] == [0, 1, 2]
    assert [urevent["latency"] for urevent in eeg["urevent"]] == [1, 13, 25]
    assert [epoch["event"] for epoch in eeg["epoch"]] == [[0], [1], [2]]

    assert len(eeg["chanlocs"]) == 128
    assert eeg["chanlocs"][0]["labels"] == "E1"
    assert len(eeg["chaninfo"]["nodatchans"]) == 4
    assert eeg["filename"] == path.name
    assert eeg["filepath"] == str(path.parent)
    assert eeg["setname"] == str(path.with_suffix(""))
    assert command == f"EEG = pop_importegimat('{path.as_posix()}', 250, 0, 'Session');"
    assert eeg["history"] == command


def test_pop_importegimat_applies_latency_offset_in_milliseconds(tmp_path: Path) -> None:
    path = tmp_path / "offset.mat"
    _segmented_fixture(path, trials=2, pnts=40)

    eeg = pop_importegimat(path, latpoint0=100, fileloc="")

    assert eeg["xmin"] == pytest.approx(-0.1)
    assert eeg["xmax"] == pytest.approx(-0.1 + 39 / 250)
    assert eeg["times"][0] == pytest.approx(-100)
    assert [event["latency"] for event in eeg["event"]] == [26, 66]
    assert len(eeg["chanlocs"]) == 128
    assert eeg["chaninfo"] == {}


def test_pop_importegimat_handles_one_millisecond_offset(tmp_path: Path) -> None:
    """Guard the exact value affected by EEGLAB's ``latpoint0 ~= 1`` typo."""
    path = tmp_path / "one_ms.mat"
    _segmented_fixture(path, trials=2)

    eeg = pop_importegimat(path, latpoint0=1, fileloc="")

    assert eeg["xmin"] == pytest.approx(-0.001)
    assert eeg["event"][0]["latency"] == pytest.approx(1.25)


def test_pop_importegimat_uses_embedded_sampling_rate(tmp_path: Path) -> None:
    path = tmp_path / "embedded_rate.mat"
    savemat(
        path,
        {
            "samplingRate": np.array([[500.0]]),
            "A_Segment1": np.ones((2, 20)),
            "A_Segment2": np.full((2, 20), 2.0),
        },
    )

    eeg, command = pop_importegimat(path, srate=128, latpoint0=20, fileloc="", return_com=True)

    assert eeg["srate"] == 500
    assert [event["latency"] for event in eeg["event"]] == [11, 31]
    assert ", 500, 20, 'Session'" in command


def test_pop_importegimat_loads_continuous_session_data(tmp_path: Path) -> None:
    path = tmp_path / "continuous.mat"
    data = np.arange(129 * 12, dtype=np.float64).reshape(129, 12)
    data[-1] = 0
    savemat(path, {"samplingRate": np.array([[1000.0]]), "Session": data})

    eeg = pop_importegimat(path, srate=250)

    np.testing.assert_array_equal(eeg["data"], data)
    assert eeg["data"].dtype == np.float64
    assert (eeg["nbchan"], eeg["pnts"], eeg["trials"]) == (129, 12, 1)
    assert eeg["srate"] == 1000
    assert eeg["setname"] == ""
    assert len(eeg["chanlocs"]) == 129
    assert len(eeg["chaninfo"]["nodatchans"]) == 3
    assert len(eeg["event"]) == 0


def test_pop_importegimat_accepts_continuous_field_prefix(tmp_path: Path) -> None:
    path = tmp_path / "continuous_prefix.mat"
    data = np.arange(8, dtype=np.float64).reshape(2, 4)
    savemat(path, {"NetStationSession": data})

    eeg, command = pop_importegimat(
        path,
        srate=200,
        data_field="NetStation",
        fileloc="",
        return_com=True,
    )

    np.testing.assert_array_equal(eeg["data"], data)
    assert eeg["srate"] == 200
    assert "'NetStation'" in command


def test_pop_importegimat_orders_types_and_numeric_segment_numbers(tmp_path: Path) -> None:
    path = tmp_path / "ordering.mat"
    savemat(
        path,
        {
            "Z_Segment100": np.full((2, 3), 100.0),
            "A_Segment10": np.full((2, 3), 10.0),
            "A_Segment2": np.full((2, 3), 2.0),
        },
    )

    eeg = pop_importegimat(path, srate=250, fileloc="")

    assert [event["type"] for event in eeg["event"]] == ["A", "A", "Z"]
    np.testing.assert_array_equal(eeg["data"][0, 0], np.array([2, 10, 100], dtype=np.float32))
    assert [event["latency"] for event in eeg["event"]] == [1, 4, 7]


def test_pop_importegimat_keeps_a_reference_channel_with_late_signal(tmp_path: Path) -> None:
    path = tmp_path / "late_reference.mat"
    segment = np.vstack((np.arange(12), np.r_[np.zeros(10), 1, 2]))
    savemat(path, {"samplingRate": [[250]], "A_Segment1": segment})

    eeg = pop_importegimat(path, fileloc="")

    np.testing.assert_array_equal(eeg["data"], segment.astype(np.float32))
    assert eeg["nbchan"] == 2


@pytest.mark.parametrize(
    ("variables", "kwargs", "message"),
    [
        ({"Session": np.ones((2, 3))}, {}, "srate is required"),
        ({"samplingRate": [[0]], "Session": np.ones((2, 3))}, {}, "srate must be positive"),
        (
            {"samplingRate": [[100], [200]], "Session": np.ones((2, 3))},
            {},
            "srate must be a scalar",
        ),
        ({"samplingRate": [[250]], "Other": np.ones((2, 3))}, {}, "data field not found"),
        (
            {"samplingRate": [[250]], "Session": np.ones((2, 3, 4))},
            {},
            "must be a 2-D",
        ),
        (
            {"samplingRate": [[250]], "A_Segment1": np.ones((2, 3)), "A_Segment2": np.ones((2, 4))},
            {},
            "same shape",
        ),
        (
            {"samplingRate": [[250]], "A_Segment1": np.ones((2, 3))},
            {"latpoint0": np.inf},
            "latpoint0 must be finite",
        ),
    ],
)
def test_pop_importegimat_rejects_invalid_inputs(
    tmp_path: Path,
    variables: dict[str, object],
    kwargs: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "invalid.mat"
    savemat(path, variables)

    with pytest.raises(ValueError, match=message):
        pop_importegimat(path, fileloc="", **kwargs)


def test_pop_importegimat_requires_an_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="EGI MATLAB file not found"):
        pop_importegimat(tmp_path / "missing.mat")


@eeglab_test(UPSTREAM_WRAPPER, "test_test_pop_importegimat")
def test_upstream_pop_importegimat_original_recording(eeglab_backend, eeglab_suite_root):
    filename = eeglab_suite_root / "unittesting_binary/testfiles/EGI/segmented_matlab.mat"
    eeg = eeglab_backend("pop_importegimat", str(filename))
    assert eeg["data"].size > 0
    assert eeg["data"].shape == (
        int(np.asarray(eeg["nbchan"]).item()),
        int(np.asarray(eeg["pnts"]).item()),
        int(np.asarray(eeg["trials"]).item()),
    )

"""BrainVision Analyzer MATLAB export import."""

import os

import numpy as np
import pytest
from scipy.io import savemat

from eegprep.functions.popfunc.pop_fileio_brainvision_mat import pop_fileio_brainvision_mat


def _channels(names, radii):
    records = np.zeros((1, len(names)), dtype=[("Name", "O"), ("Phi", "O"), ("Theta", "O"), ("Radius", "O")])
    for index, name in enumerate(names):
        records[0, index] = (name, 15.0, -20.0, radii[index])
    return records


def _markers(rows):
    columns = max(len(row) for row in rows)
    records = np.zeros(
        (len(rows), columns),
        dtype=[
            ("Description", "O"),
            ("Position", "O"),
            ("Points", "O"),
            ("Type", "O"),
            ("Chan", "O"),
            ("ChannelNumber", "O"),
        ],
    )
    for row_index, row in enumerate(rows):
        for column_index, (description, position) in enumerate(row):
            records[row_index, column_index] = (description, float(position), 1.0, "Stimulus", 0.0, 0.0)
    return records


def _write(path, payload):
    savemat(path, payload)


def _continuous_payload():
    points = 8
    names = ["Fp1", "Cz", "Pz", "Oz"]
    payload = {
        "Channels": _channels(names, [1.0, 0.0, 1.0, 1.0]),
        "SampleRate": np.array([[200.0]]),
        "SegmentCount": np.array([[1.0]]),
        "t": (np.arange(points) * 5.0).reshape(1, -1),
        "Markers": _markers([[("stim", 1), ("", 2), ("resp", 4)]]),
    }
    for index, name in enumerate(names):
        key = "xFp1" if name == "Fp1" else name
        payload[key] = (np.arange(points, dtype=float) + index).reshape(1, -1)
    return payload


def _events(eeg):
    events = eeg["event"]
    if isinstance(events, np.ndarray):
        return list(events.ravel())
    return list(events)


def test_continuous_analyzer_export_keeps_channels_rate_and_skips_blank_markers(tmp_path):
    path = tmp_path / "raw.mat"
    _write(path, _continuous_payload())
    eeg = pop_fileio_brainvision_mat(path)
    assert eeg["nbchan"] == 4
    assert eeg["pnts"] == 8
    assert eeg["trials"] == 1
    assert eeg["srate"] == 200.0
    assert eeg["data"].shape == (4, 8)
    np.testing.assert_allclose(eeg["data"][0], np.arange(8))
    np.testing.assert_allclose(eeg["data"][1], np.arange(8) + 1)
    assert [loc["labels"] for loc in eeg["chanlocs"]] == ["Fp1", "Cz", "Pz", "Oz"]
    assert "sph_theta_besa" not in eeg["chanlocs"][0]
    assert "sph_radius" in eeg["chanlocs"][0]
    assert "sph_radius" not in eeg["chanlocs"][1]
    events = _events(eeg)
    assert [event["type"] for event in events] == ["stim", "resp"]
    assert [event["latency"] for event in events] == [1.0, 4.0]
    assert [event["urevent"] for event in events] == [0, 1]
    assert eeg["filename"] == ""
    assert eeg["saved"] == "no"
    assert "pop_fileio_brainvision_mat" in eeg["history"]


def test_segmented_analyzer_export_is_channels_by_time_by_segments(tmp_path):
    points, trials = 5, 3
    names = ["Fp1", "Cz", "Pz", "Oz"]
    payload = {
        "Channels": _channels(names, [1.0, 1.0, 1.0, 1.0]),
        "SampleRate": np.array([[200.0]]),
        "SegmentCount": np.array([[float(trials)]]),
        "t": (np.arange(points) * 5.0).reshape(1, -1),
        "Markers": _markers(
            [
                [("stim", 1), ("resp", 2)],
                [("stim", 1), ("", 2)],
                [("stim", 1), ("resp", 4)],
            ]
        ),
    }
    for index, name in enumerate(names):
        payload[name] = np.full((trials, points), index, dtype=float)
    path = tmp_path / "segments.mat"
    _write(path, payload)
    eeg = pop_fileio_brainvision_mat(path)
    assert eeg["data"].shape == (4, points, trials)
    assert eeg["pnts"] == points
    assert eeg["trials"] == trials
    assert eeg["srate"] == 200.0
    np.testing.assert_allclose(eeg["data"][2, :, 1], np.full(points, 2.0))
    events = _events(eeg)
    assert len(events) == 5
    assert [event["epoch"] for event in events] == [1, 1, 2, 3, 3]
    assert [event["latency"] for event in events] == [1.0, 2.0, 6.0, 11.0, 14.0]


def test_exact_channel_name_wins_over_the_prefix_match(tmp_path):
    payload = {
        "Channels": _channels(["p1", "Fp1"], [0.0, 0.0]),
        "SampleRate": np.array([[100.0]]),
        "SegmentCount": np.array([[1.0]]),
        "t": np.arange(4, dtype=float).reshape(1, -1),
        "Fp1": np.array([[1.0, 2.0, 3.0, 4.0]]),
    }
    path = tmp_path / "prefix.mat"
    _write(path, payload)
    eeg = pop_fileio_brainvision_mat(path)
    np.testing.assert_allclose(eeg["data"][0], np.zeros(4))
    np.testing.assert_allclose(eeg["data"][1], [1, 2, 3, 4])


def test_unknown_channel_variable_is_appended_once(tmp_path, caplog):
    payload = _continuous_payload()
    payload["EOG"] = np.arange(8, dtype=float).reshape(1, -1)
    path = tmp_path / "extra.mat"
    _write(path, payload)
    with caplog.at_level("WARNING"):
        eeg = pop_fileio_brainvision_mat(path)
    assert [loc["labels"] for loc in eeg["chanlocs"]] == ["Fp1", "Cz", "Pz", "Oz", "EOG"]
    assert eeg["nbchan"] == 5
    assert "eog" in caplog.text.lower()


def test_channel_shape_mismatch_is_an_error(tmp_path):
    payload = {
        "Channels": _channels(["Fp1", "Cz"], [0.0, 0.0]),
        "SampleRate": np.array([[100.0]]),
        "SegmentCount": np.array([[1.0]]),
        "t": np.arange(4, dtype=float),
        "Fp1": np.arange(4, dtype=float),
        "Cz": np.arange(5, dtype=float),
    }
    path = tmp_path / "mismatch.mat"
    _write(path, payload)
    with pytest.raises(ValueError, match="do not share a shape") as caught:
        pop_fileio_brainvision_mat(path)
    assert "pop_fileio_brainvision_mat" in str(caught.value)
    assert str(path) in str(caught.value)


def test_plain_numeric_mat_is_not_imported_as_one_channel(tmp_path):
    path = tmp_path / "plain.mat"
    savemat(path, {"data": np.zeros((1, 10))})
    with pytest.raises(ValueError, match="Channels") as caught:
        pop_fileio_brainvision_mat(path)
    message = str(caught.value)
    assert "pop_fileio_brainvision_mat" in message
    assert str(path) in message
    assert "pop_loadbv" in message


def test_matlab_v73_file_is_refused(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "v73.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("SampleRate", data=np.array([200.0]))
    with pytest.raises(ValueError, match="v7.3") as caught:
        pop_fileio_brainvision_mat(path)
    assert "pop_fileio_brainvision_mat" in str(caught.value)
    assert str(path) in str(caught.value)


def test_menu_import_uses_the_analyzer_layout(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from unittest import mock

    from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
    from eegprep.functions.guifunc.session import EEGPrepSession

    path = tmp_path / "menu.mat"
    _write(path, _continuous_payload())
    session = EEGPrepSession()

    class QFileDialog:
        @staticmethod
        def getOpenFileName(*_args, **_kwargs):
            return str(path), ""

    qt_widgets = type("FakeQtWidgets", (), {"QFileDialog": QFileDialog})
    dispatcher = MenuActionDispatcher(session)
    with mock.patch("eegprep.functions.guifunc.menu_actions._require_qt_widgets", return_value=qt_widgets):
        dispatcher.dispatch("pop_fileio_brainvision_mat")
    eeg = session.EEG
    assert eeg["data"].shape == (4, 8)
    assert eeg["srate"] == 200.0
    assert len(_events(eeg)) == 2
    assert eeg["filename"] == ""
    assert path.is_file()
    assert "pop_fileio_brainvision_mat" in session.LASTCOM

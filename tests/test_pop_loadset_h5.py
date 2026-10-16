"""Tests for MATLAB v7.3 (HDF5) .set loading."""

from pathlib import Path
import unittest

import h5py
import numpy as np
import pytest

from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5

_MATLAB_CASES = ("sample", "no_events", "one_channel", "square", "epochs_ica")
_MUTATIONS = {
    "sample": "",
    "no_events": "EEG.event = []; EEG.urevent = [];",
    "one_channel": "EEG = pop_select(EEG, 'channel', 1);",
    "square": "n = min(EEG.nbchan, EEG.pnts); EEG = pop_select(EEG, 'channel', 1:n, 'point', [1 n]);",
    "epochs_ica": "",
}
_IGNORED_FIELDS = {"filename", "datfile"}


class TestPopLoadsetH5RealData(unittest.TestCase):
    """Compare committed v7.3 sample sets with the v7 save of the same recording."""

    def test_compare_continuous_pop_loadset(self):
        v7 = pop_loadset("sample_data/eeglab_data.set")
        h5 = pop_loadset_h5("sample_data/eeglab_data_hdf5.set")

        self.assertEqual(h5["chanlocs"][0]["labels"], "FPz")
        self.assertEqual(h5["event"][0]["type"], "square")
        self.assertEqual(_labels(h5), _labels(v7))
        self.assertEqual(_event_types(h5), _event_types(v7))
        np.testing.assert_array_equal(h5["data"], v7["data"])
        summary = eeg_compare(v7, h5)
        self.assertNotIn("channel label(s) differ", summary)
        self.assertNotIn("channel_differences", summary)
        self.assertNotIn("NOT OK", summary)

    def test_compare_epochs_pop_loadset(self):
        v7 = pop_loadset("sample_data/eeglab_data_epochs_ica.set")
        h5 = pop_loadset_h5("sample_data/eeglab_data_epochs_ica_hdf5.set")

        self.assertEqual(_labels(h5), _labels(v7))
        self.assertEqual(_event_types(h5), _event_types(v7))
        self.assertEqual(h5["data"].shape, (32, 384, 80))
        np.testing.assert_array_equal(h5["data"], v7["data"])
        np.testing.assert_array_equal(h5["icaweights"], v7["icaweights"])
        np.testing.assert_array_equal(h5["icasphere"], v7["icasphere"])
        np.testing.assert_array_equal(h5["icawinv"], v7["icawinv"])
        summary = eeg_compare(v7, h5)
        self.assertNotIn("channel label(s) differ", summary)
        self.assertNotIn("channel_differences", summary)
        self.assertNotIn("NOT OK", summary)

    def test_direct_call_returns_zero_based_urevent_and_urchan(self):
        direct = pop_loadset_h5("sample_data/eeglab_data_hdf5.set")
        via_loadset = pop_loadset("sample_data/eeglab_data_hdf5.set")
        from_set = pop_loadset("sample_data/eeglab_data.set")

        urevents = [int(event["urevent"]) for event in direct["event"]]
        self.assertEqual(urevents, list(range(len(direct["urevent"]))))
        self.assertEqual(urevents, [int(event["urevent"]) for event in via_loadset["event"]])
        self.assertEqual(urevents, [int(event["urevent"]) for event in from_set["event"]])
        urchans = [int(chan["urchan"]) for chan in direct["chanlocs"]]
        self.assertEqual(urchans, list(range(direct["nbchan"])))
        self.assertEqual(urchans, [int(chan["urchan"]) for chan in via_loadset["chanlocs"]])
        self.assertEqual(urchans, [int(chan["urchan"]) for chan in from_set["chanlocs"]])


def test_h5_square_data_uses_reversed_dimensions(tmp_path):
    matlab = np.array([[1, 2], [3, 4]], dtype=np.float32)
    path = tmp_path / "square.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        for name, value in (("nbchan", 2), ("pnts", 2), ("trials", 1), ("srate", 1), ("xmin", 0), ("xmax", 1)):
            _double_scalar(eeg, name, value)
        data = eeg.create_dataset("data", data=np.transpose(matlab))
        data.attrs["MATLAB_class"] = "single"

    loaded = pop_loadset_h5(str(path))

    assert loaded["data"].shape == (2, 2)
    np.testing.assert_array_equal(loaded["data"], matlab)


def test_h5_one_by_one_struct_label_and_empty_event(tmp_path):
    samples = np.arange(5, dtype=np.float32).reshape(1, 5)
    path = tmp_path / "one.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        _double_scalar(eeg, "nbchan", 1)
        _double_scalar(eeg, "pnts", 5)
        _double_scalar(eeg, "trials", 1)
        _double_scalar(eeg, "srate", 100)
        _double_scalar(eeg, "xmin", 0)
        _double_scalar(eeg, "xmax", 0.04)
        data = eeg.create_dataset("data", data=np.transpose(samples))
        data.attrs["MATLAB_class"] = "single"
        event = eeg.create_dataset("event", data=np.zeros(2, dtype=np.uint64))
        event.attrs["MATLAB_class"] = "double"
        event.attrs["MATLAB_empty"] = np.uint8(1)
        chanlocs = eeg.create_group("chanlocs")
        chanlocs.attrs["MATLAB_class"] = "struct"
        refs = chanlocs.create_dataset("labels", shape=(1, 1), dtype=h5py.ref_dtype)
        label = handle.create_dataset("#refs#/label", data=np.array([[ord("F")], [ord("z")]], dtype=np.uint16))
        label.attrs["MATLAB_class"] = "char"
        label.attrs["MATLAB_int_decode"] = np.int32(2)
        refs[0, 0] = label.ref

    loaded = pop_loadset_h5(str(path))

    assert loaded["chanlocs"][0]["labels"] == "Fz"
    assert len(loaded["event"]) == 0
    assert loaded["data"].shape == (1, 5)
    np.testing.assert_array_equal(loaded["data"], samples)


@pytest.mark.parametrize("case", _MATLAB_CASES)
def test_matlab_v73_matches_v7_and_matlab_reload(eeglab_matlab_engine, tmp_path, case):
    source = _sample_set(case)
    v7_path, v73_path = _write_matlab_pair(eeglab_matlab_engine, tmp_path, source, case)

    from_v7 = pop_loadset(str(v7_path))
    from_v73 = pop_loadset(str(v73_path))
    _assert_same_eeg(from_v73, from_v7)
    _assert_case_shape(case, from_v73)
    _assert_matches_matlab_reload(eeglab_matlab_engine, v7_path, v73_path, from_v73)


def _sample_set(case):
    name = "eeglab_data_epochs_ica.set" if case == "epochs_ica" else "eeglab_data.set"
    return Path("sample_data", name).resolve()


def _write_matlab_pair(engine, directory, source, case):
    v7_path = directory / f"{case}_v7.set"
    v73_path = directory / f"{case}_v73.set"
    source_m = str(source).replace("'", "''")
    directory_m = str(directory).replace("'", "''")
    engine.eval(
        f"EEG = pop_loadset('{source_m}');"
        f"{_MUTATIONS[case]}"
        f"pop_saveset(EEG, 'filename', '{v7_path.name}', 'filepath', '{directory_m}', "
        f"'savemode', 'onefile', 'version', '7');"
        f"pop_saveset(EEG, 'filename', '{v73_path.name}', 'filepath', '{directory_m}', "
        f"'savemode', 'onefile', 'version', '7.3');",
        nargout=0,
    )
    return v7_path, v73_path


def _assert_case_shape(case, eeg):
    if case == "no_events":
        assert len(eeg["event"]) == 0
    if case == "one_channel":
        assert int(eeg["nbchan"]) == 1
        assert eeg["data"].shape[0] == 1
        assert isinstance(eeg["chanlocs"][0]["labels"], str)
        assert len(eeg["chanlocs"][0]["labels"]) > 1
    if case == "square":
        assert eeg["data"].shape[0] == eeg["data"].shape[1] == int(eeg["nbchan"])
    if case == "epochs_ica":
        assert eeg["data"].ndim == 3
        assert eeg["icaweights"].size > 0


def _assert_matches_matlab_reload(engine, v7_path, v73_path, python_eeg):
    v7_m = str(v7_path).replace("'", "''")
    v73_m = str(v73_path).replace("'", "''")
    engine.eval(
        f"A = pop_loadset('{v7_m}');"
        f"B = pop_loadset('{v73_m}');"
        "A.filename = ''; B.filename = '';"
        "A.filepath = ''; B.filepath = '';"
        "A.datfile = ''; B.datfile = '';"
        "matlab_same = isequaln(A, B);",
        nargout=0,
    )
    assert bool(engine.eval("matlab_same", nargout=1))
    matlab_data = np.array(engine.eval("B.data", nargout=1))
    actual = np.asarray(python_eeg["data"])
    assert actual.shape == matlab_data.shape
    np.testing.assert_array_equal(actual, np.asarray(matlab_data, dtype=actual.dtype))
    if len(python_eeg["chanlocs"]):
        assert list(engine.eval("{B.chanlocs.labels}", nargout=1)) == _labels(python_eeg)
    n_events = int(np.squeeze(engine.eval("numel(B.event)", nargout=1)))
    assert n_events == len(python_eeg["event"])
    if n_events:
        assert list(engine.eval("{B.event.type}", nargout=1)) == _event_types(python_eeg)


def _assert_same_eeg(left, right):
    assert set(left) == set(right)
    for key in left:
        if key in _IGNORED_FIELDS:
            continue
        _assert_same(left[key], right[key], key)


def _assert_same(left, right, path):
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict), path
        assert set(left) == set(right), path
        for key in left:
            _assert_same(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert left_array.shape == right_array.shape, (path, left_array.shape, right_array.shape)
        if left_array.dtype == object or right_array.dtype == object:
            for index in np.ndindex(left_array.shape):
                _assert_same(left_array[index], right_array[index], f"{path}{index}")
            return
        np.testing.assert_array_equal(left_array, right_array, err_msg=path)
        return
    assert left == right, (path, left, right)


def _labels(eeg):
    return [channel["labels"] for channel in eeg["chanlocs"]]


def _event_types(eeg):
    events = eeg.get("event")
    if events is None or len(events) == 0:
        return []
    return [event["type"] for event in events]


def _double_scalar(group, name, value):
    field = group.create_dataset(name, data=np.array([[value]], dtype=np.float64))
    field.attrs["MATLAB_class"] = "double"
    return field


if __name__ == "__main__":
    unittest.main()

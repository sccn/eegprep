"""Tests for MATLAB v7.3 (HDF5) .set loading."""

from pathlib import Path
import unittest

import h5py
import numpy as np
import pytest
import scipy.io

from eegprep.functions.popfunc.eeg_compare import eeg_compare
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5

_MATLAB_CASES = ("sample", "no_events", "one_channel", "square", "epochs_ica", "twofiles", "empty_time")
_MUTATIONS = {
    "sample": "",
    "no_events": "EEG.event = []; EEG.urevent = [];",
    "one_channel": "EEG = pop_select(EEG, 'channel', 1);",
    "square": "n = min(EEG.nbchan, EEG.pnts); EEG = pop_select(EEG, 'channel', 1:n, 'point', [1 n]);",
    "epochs_ica": "",
    "twofiles": "",
    "empty_time": "EEG = pop_select(EEG, 'point', [1 1]);",
}
_IGNORED_FIELDS = {"filename", "datfile"}
_V7_SHAPE_FIELDS = {
    "one_channel": ("data",),
    "empty_time": ("data",),
    "no_events": ("event", "urevent"),
}
_ICA_FIELDS = ("icaweights", "icasphere", "icawinv", "icaact")
_CHAN_FIELDS = ("theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius")


class TestPopLoadsetH5RealData(unittest.TestCase):
    """Compare committed v7.3 sample sets with the v7 save of the same recording."""

    def test_compare_continuous_pop_loadset(self):
        v7 = pop_loadset("sample_data/eeglab_data.set")
        h5 = pop_loadset("sample_data/eeglab_data_hdf5.set")

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
        h5 = pop_loadset("sample_data/eeglab_data_epochs_ica_hdf5.set")

        self.assertEqual(_labels(h5), _labels(v7))
        self.assertEqual(_event_types(h5), _event_types(v7))
        self.assertEqual(h5["data"].shape, (32, 384, 80))
        np.testing.assert_array_equal(h5["data"], v7["data"])
        # The v7 sample was saved with icaact cleared, so its load scales.
        # The v7.3 sample still stores icaact, so its load keeps the saved weights.
        stored = scipy.io.loadmat(
            "sample_data/eeglab_data_epochs_ica.set",
            struct_as_record=False,
            squeeze_me=True,
            appendmat=False,
        )["EEG"]
        raw = pop_loadset_h5("sample_data/eeglab_data_epochs_ica_hdf5.set")
        np.testing.assert_array_equal(raw["icaweights"], stored.icaweights)
        np.testing.assert_array_equal(raw["icasphere"], stored.icasphere)
        np.testing.assert_array_equal(raw["icawinv"], stored.icawinv)
        self.assertEqual(raw["icaweights"].ndim, 2)
        self.assertEqual(raw["icaact"].shape, (32, 384, 80))
        summary = eeg_compare(v7, h5)
        self.assertNotIn("channel label(s) differ", summary)
        self.assertNotIn("channel_differences", summary)
        self.assertNotIn("NOT OK", summary)

    def test_direct_call_returns_zero_based_urevent_and_urchan(self):
        direct = pop_loadset("sample_data/eeglab_data_hdf5.set")
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

    loaded = pop_loadset(str(path))

    assert loaded["chanlocs"][0]["labels"] == "Fz"
    assert loaded["event"].shape == (0, 0)
    assert len(loaded["event"]) == 0
    assert loaded["data"].shape == (1, 5)
    np.testing.assert_array_equal(loaded["data"], samples)


@pytest.mark.parametrize("case", _MATLAB_CASES)
def test_matlab_v73_matches_v7_and_matlab_reload(eeglab_matlab_engine, tmp_path, case):
    source = _sample_set(case)
    v7_path, v73_path = _write_matlab_pair(eeglab_matlab_engine, tmp_path, source, case)

    from_v7 = pop_loadset(str(v7_path))
    from_v73 = pop_loadset(str(v73_path))
    _assert_same_eeg(from_v73, from_v7, _V7_SHAPE_FIELDS.get(case, ()))
    _assert_case_shape(case, from_v73)
    if case == "twofiles":
        assert v7_path.with_suffix(".fdt").is_file()
        assert v73_path.with_suffix(".fdt").is_file()
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
        f"'savemode', '{_savemode(case)}', 'version', '7');"
        f"pop_saveset(EEG, 'filename', '{v73_path.name}', 'filepath', '{directory_m}', "
        f"'savemode', '{_savemode(case)}', 'version', '7.3');",
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
        assert eeg["icaweights"].ndim == 2
        assert eeg["icaweights"].size > 0
    if case == "empty_time":
        assert eeg["data"].shape == (int(eeg["nbchan"]), 0)
    if case == "twofiles":
        assert eeg["data"].shape[0] == int(eeg["nbchan"])


def _savemode(case):
    return "twofiles" if case == "twofiles" else "onefile"


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
    _assert_matlab_array(engine, "B.data", python_eeg["data"])
    for field in _ICA_FIELDS:
        _assert_matlab_field(engine, f"B.{field}", python_eeg.get(field))
    etc = python_eeg.get("etc") if isinstance(python_eeg.get("etc"), dict) else {}
    for field in ("icaweights_beforerms", "icasphere_beforerms"):
        if field in etc:
            _assert_matlab_field(engine, f"B.etc.{field}", etc[field])
    _assert_matlab_chanlocs(engine, python_eeg)
    _assert_matlab_epoch(engine, python_eeg)
    n_events = int(np.squeeze(engine.eval("numel(B.event)", nargout=1)))
    assert n_events == len(python_eeg["event"])
    if n_events:
        assert list(engine.eval("{B.event.type}", nargout=1)) == _event_types(python_eeg)


def _assert_matlab_field(engine, expr, value):
    if not isinstance(value, np.ndarray) or value.size == 0:
        assert int(np.squeeze(engine.eval(f"numel({expr})", nargout=1))) == 0
        return
    _assert_matlab_array(engine, expr, value)


def _assert_matlab_array(engine, expr, actual):
    matlab_data = np.array(engine.eval(expr, nargout=1))
    actual = np.asarray(actual)
    assert actual.shape == tuple(matlab_data.shape), (expr, actual.shape, matlab_data.shape)
    np.testing.assert_array_equal(actual, np.asarray(matlab_data, dtype=actual.dtype))


def _assert_matlab_chanlocs(engine, python_eeg):
    chanlocs = python_eeg.get("chanlocs")
    count = 0 if chanlocs is None else len(chanlocs)
    assert int(np.squeeze(engine.eval("numel(B.chanlocs)", nargout=1))) == count
    if count == 0:
        return
    assert list(engine.eval("{B.chanlocs.labels}", nargout=1)) == _labels(python_eeg)
    for field in _CHAN_FIELDS:
        if field not in chanlocs[0]:
            continue
        matlab_vals = np.asarray(engine.eval(f"[B.chanlocs.{field}]", nargout=1), dtype=float).reshape(-1)
        python_vals = np.array([_number(channel.get(field)) for channel in chanlocs], dtype=float)
        np.testing.assert_array_equal(python_vals, matlab_vals, err_msg=field)
    if "urchan" in chanlocs[0]:
        matlab_ur = np.asarray(engine.eval("[B.chanlocs.urchan]", nargout=1), dtype=float).reshape(-1)
        python_ur = np.array([channel["urchan"] for channel in chanlocs], dtype=float)
        np.testing.assert_array_equal(python_ur, matlab_ur - 1)
    if "type" in chanlocs[0]:
        assert list(engine.eval("{B.chanlocs.type}", nargout=1)) == [channel.get("type") for channel in chanlocs]


def _assert_matlab_epoch(engine, python_eeg):
    epoch = python_eeg.get("epoch")
    count = 0 if epoch is None else len(epoch)
    assert int(np.squeeze(engine.eval("numel(B.epoch)", nargout=1))) == count
    if count == 0:
        return
    engine.eval(
        "epoch_event = arrayfun(@(e) e.event, B.epoch, 'UniformOutput', false);"
        "epoch_urevent = arrayfun(@(e) e.eventurevent, B.epoch, 'UniformOutput', false);"
        "epoch_latency = arrayfun(@(e) e.eventlatency, B.epoch, 'UniformOutput', false);"
        "epoch_type = arrayfun(@(e) e.eventtype, B.epoch, 'UniformOutput', false);"
        "epoch_position = arrayfun(@(e) e.eventposition, B.epoch, 'UniformOutput', false);",
        nargout=0,
    )
    _assert_epoch_vectors(engine, "epoch_event", epoch, "event", offset=1)
    _assert_epoch_vectors(engine, "epoch_urevent", epoch, "eventurevent", offset=1)
    _assert_epoch_vectors(engine, "epoch_latency", epoch, "eventlatency", offset=0)
    _assert_epoch_vectors(engine, "epoch_position", epoch, "eventposition", offset=0)
    matlab_types = engine.eval("epoch_type", nargout=1)
    for index, record in enumerate(epoch):
        assert list(np.ravel(matlab_types[index])) == list(record["eventtype"])


def _assert_epoch_vectors(engine, matlab_name, epoch, field, offset):
    rows = engine.eval(matlab_name, nargout=1)
    for index, record in enumerate(epoch):
        matlab_vals = np.asarray(rows[index], dtype=float).reshape(-1)
        python_vals = np.array([_number(value) for value in record[field]], dtype=float)
        np.testing.assert_array_equal(python_vals, matlab_vals - offset)


def _number(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        return float(np.reshape(value, -1)[0])
    return float(value)


def _assert_same_eeg(left, right, ignored=()):
    ignored = _IGNORED_FIELDS | set(ignored)
    assert set(left) == set(right)
    for key in left:
        if key in ignored:
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


def test_h5_empty_keeps_reversed_dimensions(tmp_path):
    path = tmp_path / "empty.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        data = eeg.create_dataset("data", data=np.array([0, 32], dtype=np.uint64))
        data.attrs["MATLAB_class"] = "single"
        data.attrs["MATLAB_empty"] = np.uint8(1)

    loaded = pop_loadset_h5(str(path))

    assert loaded["data"].shape == (32, 0)
    assert loaded["data"].dtype == np.float32


def test_h5_complex_decodes_real_and_imag(tmp_path):
    values = np.array([1 + 2j, 3 - 4j], dtype=np.complex128)
    stored = np.empty(np.transpose(values.reshape(1, -1)).shape, dtype=[("real", np.float64), ("imag", np.float64)])
    transposed = np.transpose(values.reshape(1, -1))
    stored["real"] = transposed.real
    stored["imag"] = transposed.imag
    path = tmp_path / "complex.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        etc = eeg.create_group("etc")
        cplx = etc.create_dataset("cplx", data=stored)
        cplx.attrs["MATLAB_class"] = "double"

    loaded = pop_loadset_h5(str(path))

    np.testing.assert_array_equal(loaded["etc"]["cplx"], values)


def test_h5_sparse_decodes_as_csc(tmp_path):
    path = tmp_path / "sparse.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        matrix = eeg.create_group("etc")
        sparse_group = matrix.create_group("weights")
        sparse_group.attrs["MATLAB_class"] = "double"
        sparse_group.attrs["MATLAB_sparse"] = np.uint64(3)
        _vector(sparse_group, "data", [1, 2, 3], np.float64)
        _vector(sparse_group, "ir", [0, 2, 1], np.uint64)
        _vector(sparse_group, "jc", [0, 1, 2, 2, 3], np.uint64)

    loaded = pop_loadset_h5(str(path))

    assert loaded["etc"]["weights"].shape == (3, 4)
    np.testing.assert_array_equal(
        loaded["etc"]["weights"].toarray(),
        np.array([[1, 0, 0, 0], [0, 0, 0, 3], [0, 2, 0, 0]], dtype=float),
    )


def test_h5_one_component_icaweights_stay_two_dimensional(tmp_path):
    weights = np.array([[1.0, 2.0, 3.0, 4.0]])
    path = tmp_path / "ica.set"
    with h5py.File(path, "w") as handle:
        eeg = handle.create_group("EEG")
        dataset = eeg.create_dataset("icaweights", data=np.transpose(weights))
        dataset.attrs["MATLAB_class"] = "double"

    loaded = pop_loadset_h5(str(path))

    assert loaded["icaweights"].shape == (1, 4)
    np.testing.assert_array_equal(loaded["icaweights"], weights)


def _vector(group, name, values, dtype):
    column = np.asarray(values, dtype=dtype).reshape(-1, 1)
    dataset = group.create_dataset(name, data=np.transpose(column))
    dataset.attrs["MATLAB_class"] = "double" if dtype == np.float64 else "uint64"
    return dataset


if __name__ == "__main__":
    unittest.main()

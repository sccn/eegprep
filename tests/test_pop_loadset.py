import h5py
import numpy as np
import pytest
import scipy.io

from eegprep.functions.popfunc._matlab_fields import apply_field_rules
from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.functions.popfunc.pop_select import pop_select


def test_pop_loadset_marks_loaded_dataset_justloaded():
    eeg = pop_loadset("sample_data/eeglab_data.set")

    assert eeg["saved"] == "justloaded"


def test_pop_loadset_keeps_embedded_data_when_datfile_sidecar_is_missing(tmp_path):
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    set_file = tmp_path / "embedded.set"
    scipy.io.savemat(
        set_file,
        {
            "setname": "embedded",
            "filename": set_file.name,
            "filepath": str(tmp_path),
            "nbchan": 2.0,
            "pnts": 3.0,
            "trials": 1.0,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": 0.02,
            "data": data,
            "datfile": "stale.fdt",
        },
        appendmat=False,
    )

    loaded = pop_loadset(set_file)
    info = pop_loadset(set_file, loadmode="info")

    np.testing.assert_array_equal(loaded["data"], data)
    np.testing.assert_array_equal(info["data"], data)
    assert loaded["nbchan"] == 2
    assert loaded["pnts"] == 3

    np.zeros(data.size, dtype=np.float32).tofile(tmp_path / "stale.fdt")
    reloaded = pop_loadset(set_file)
    np.testing.assert_array_equal(reloaded["data"], data)


def test_pop_loadset_normalizes_nonempty_icachansind_to_zero_based_integers():
    eeg = pop_loadset("sample_data/eeglab_data_epochs_ica.set")

    assert np.issubdtype(eeg["icachansind"].dtype, np.integer)
    np.testing.assert_array_equal(eeg["icachansind"][:5], np.array([0, 1, 2, 3, 4]))


@pytest.mark.parametrize(
    "path", ["sample_data/eeglab_data_epochs_ica.set", "sample_data/eeglab_data_epochs_ica_hdf5.set"]
)
def test_pop_loadset_epoch_eventurevent_is_zero_based_like_event_urevent(path):
    eeg = pop_loadset(path)

    assert list(eeg["epoch"][0]["event"]) == [0, 1, 2]
    assert list(eeg["epoch"][0]["eventurevent"]) == [0, 1, 2]
    for ep in eeg["epoch"]:
        assert list(ep["eventurevent"]) == [eeg["event"][i]["urevent"] for i in ep["event"]]


def _singleton_eeg(data: np.ndarray) -> dict:
    eeg = eeg_emptyset()
    trials = int(data.shape[2]) if data.ndim == 3 else 1
    pnts = int(data.shape[1])
    eeg.update(
        {
            "setname": "singleton",
            "data": np.array(data, dtype=np.float64, copy=True),
            "nbchan": int(data.shape[0]),
            "pnts": pnts,
            "trials": trials,
            "srate": 100.0,
            "xmin": 0.0,
            "xmax": (pnts - 1) / 100.0,
            "times": np.arange(pnts, dtype=float),
            "chanlocs": [{"labels": f"Ch{index}"} for index in range(1, int(data.shape[0]) + 1)],
            "event": [],
            "saved": "no",
        }
    )
    return eeg


def _singleton_case(kind: str) -> tuple[dict, dict[str, tuple[int, ...]]]:
    if kind == "one_channel":
        data = np.arange(16, dtype=np.float64).reshape(1, 16)
        return _singleton_eeg(data), {}
    if kind == "one_channel_epochs":
        data = np.arange(5 * 3, dtype=np.float64).reshape(1, 5, 3)
        return _singleton_eeg(data), {}
    if kind == "one_sample":
        data = np.arange(6, dtype=np.float64).reshape(6, 1)
        return _singleton_eeg(data), {}
    if kind == "one_component":
        nchan = 4
        data = np.arange(nchan * 8, dtype=np.float64).reshape(nchan, 8)
        weights = np.array([[0.25, -0.5, 0.125, 0.125]], dtype=np.float64)
        sphere = np.eye(nchan, dtype=np.float64)
        eeg = _singleton_eeg(data)
        eeg["icaweights"] = weights
        eeg["icasphere"] = sphere
        eeg["icawinv"] = np.linalg.pinv(weights @ sphere)
        eeg["icachansind"] = np.arange(nchan, dtype=int)
        eeg["icaact"] = weights @ sphere @ data
        return eeg, {
            "icaweights": (1, nchan),
            "icasphere": (nchan, nchan),
            "icawinv": (nchan, 1),
        }
    if kind == "one_ica_channel":
        data = np.arange(8, dtype=np.float64).reshape(1, 8)
        weights = np.array([[0.5]], dtype=np.float64)
        sphere = np.array([[1.0]], dtype=np.float64)
        eeg = _singleton_eeg(data)
        eeg["icaweights"] = weights
        eeg["icasphere"] = sphere
        eeg["icawinv"] = np.linalg.pinv(weights @ sphere)
        eeg["icachansind"] = np.array([0])
        eeg["icaact"] = weights @ sphere @ data
        return eeg, {
            "icaweights": (1, 1),
            "icasphere": (1, 1),
            "icawinv": (1, 1),
        }
    raise AssertionError(kind)


@pytest.mark.parametrize("savemode", ["onefile", "twofiles"])
@pytest.mark.parametrize(
    "kind", ["one_channel", "one_channel_epochs", "one_sample", "one_component", "one_ica_channel"]
)
def test_pop_loadset_roundtrip_keeps_singleton_dimensions(tmp_path, savemode, kind):
    eeg, ica_shapes = _singleton_case(kind)
    expected = np.array(eeg["data"], copy=True)
    saved = pop_saveset(eeg, "filename", f"{kind}.set", "filepath", tmp_path, "savemode", savemode)

    loaded = pop_loadset(saved["filename"], "filepath", saved["filepath"])

    assert loaded["data"].shape == expected.shape
    np.testing.assert_array_equal(loaded["data"], expected)
    assert loaded["nbchan"] == expected.shape[0]
    assert loaded["pnts"] == expected.shape[1]
    for field, shape in ica_shapes.items():
        assert isinstance(loaded[field], np.ndarray)
        assert loaded[field].shape == shape
        assert loaded[field].ndim == 2


def _matlab_shape(engine, expression: str) -> tuple[int, ...]:
    values = np.array(engine.eval(f"size({expression})", nargout=1), dtype=int).reshape(-1)
    return tuple(int(value) for value in values)


_MATLAB_SINGLETON_SETUP = {
    "one_channel": """
        EEG = eeg_emptyset;
        EEG.nbchan = 1;
        EEG.pnts = 16;
        EEG.trials = 1;
        EEG.srate = 128;
        EEG.xmin = 0;
        EEG.xmax = 15/128;
        EEG.data = reshape(double(1:16), 1, 16);
        EEG.chanlocs = struct('labels', 'Cz', 'X', 1, 'Y', 0, 'Z', 0);
        EEG.event = struct('type', 'square', 'latency', 4, 'urevent', 1);
        EEG.urevent = struct('type', 'square', 'latency', 4);
    """,
    "one_channel_epochs": """
        EEG = eeg_emptyset;
        EEG.nbchan = 1;
        EEG.pnts = 5;
        EEG.trials = 3;
        EEG.srate = 128;
        EEG.xmin = 0;
        EEG.xmax = 4/128;
        EEG.data = reshape(double(1:(5*3)), 1, 5, 3);
        EEG.chanlocs = struct('labels', 'Cz', 'X', 1, 'Y', 2, 'Z', 3);
        EEG.event = struct('type', {'square','rt','square'}, 'latency', {1, 6, 11}, 'urevent', {1, 2, 3}, 'epoch', {1, 2, 3});
        EEG.urevent = struct('type', {'square','rt','square'}, 'latency', {1, 6, 11});
        EEG = eeg_checkset(EEG, 'eventconsistency');
    """,
    "one_sample": """
        EEG = eeg_emptyset;
        EEG.nbchan = 6;
        EEG.pnts = 1;
        EEG.trials = 1;
        EEG.srate = 128;
        EEG.xmin = 0;
        EEG.xmax = 0;
        EEG.data = double((1:6)');
        EEG.chanlocs = struct('labels', {'E1','E2','E3','E4','E5','E6'});
    """,
    "one_component": """
        EEG = eeg_emptyset;
        EEG.nbchan = 4;
        EEG.pnts = 8;
        EEG.trials = 1;
        EEG.srate = 128;
        EEG.xmin = 0;
        EEG.xmax = 7/128;
        EEG.data = reshape(double(1:(4*8)), 4, 8);
        EEG.icaweights = [0.25 -0.5 0.125 0.125];
        EEG.icasphere = eye(4);
        EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
        EEG.icachansind = 1:4;
        EEG.icaact = (EEG.icaweights * EEG.icasphere) * EEG.data;
        EEG.chanlocs = struct('labels', {'E1','E2','E3','E4'});
    """,
    "one_ica_channel": """
        EEG = eeg_emptyset;
        EEG.nbchan = 1;
        EEG.pnts = 8;
        EEG.trials = 1;
        EEG.srate = 128;
        EEG.xmin = 0;
        EEG.xmax = 7/128;
        EEG.data = reshape(double(1:8), 1, 8);
        EEG.icaweights = 0.5;
        EEG.icasphere = 1;
        EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
        EEG.icachansind = 1;
        EEG.icaact = (EEG.icaweights * EEG.icasphere) * EEG.data;
        EEG.chanlocs = struct('labels', 'Cz', 'X', 1, 'Y', 0, 'Z', 0);
        EEG.event = struct('type', 'square', 'latency', 4, 'urevent', 1);
        EEG.urevent = struct('type', 'square', 'latency', 4);
    """,
}


@pytest.mark.parametrize("savemode", ["onefile", "twofiles"])
@pytest.mark.parametrize(
    "kind", ["one_channel", "one_channel_epochs", "one_sample", "one_component", "one_ica_channel"]
)
def test_matlab_pop_loadset_singleton_shapes_match_matlab(eeglab_matlab_engine, tmp_path, savemode, kind):
    """MATLAB writes each singleton case. EEGPrep's shape matches size() and the data matches."""
    folder = tmp_path.as_posix().replace("'", "''")
    filename = f"{kind}_{savemode}.set"
    # v7 stays on scipy loadmat. pop_saveset's default v7.3 is HDF5.
    eeglab_matlab_engine.eval(
        _MATLAB_SINGLETON_SETUP[kind]
        + f"""
        EEG = pop_saveset(EEG, 'filename', '{filename}', 'filepath', '{folder}', 'savemode', '{savemode}', 'version', '7');
        ML = pop_loadset('{filename}', '{folder}');
        """,
        nargout=0,
    )
    matlab_shape = _matlab_shape(eeglab_matlab_engine, "ML.data")
    matlab_data = np.array(eeglab_matlab_engine.eval("ML.data", nargout=1))

    loaded = pop_loadset(tmp_path / filename)

    assert loaded["data"].shape == matlab_shape
    np.testing.assert_array_equal(np.asarray(loaded["data"]), matlab_data)
    # pop_saveset.m:225-226 stores icaact=[], so both sides recompute icawinv with pinv.
    ica_fields = (
        ("icaweights", "icasphere", "icaact")
        if kind == "one_component"
        else (
            "icaweights",
            "icasphere",
            "icawinv",
            "icaact",
        )
    )
    for field in ica_fields:
        _assert_ica_field(eeglab_matlab_engine, loaded, field)
    _assert_times(eeglab_matlab_engine, loaded)
    _assert_chanlocs(eeglab_matlab_engine, loaded)
    _assert_events(eeglab_matlab_engine, loaded)
    _assert_epochs(eeglab_matlab_engine, loaded)


def _assert_ica_field(engine, loaded, field: str) -> None:
    shape = _matlab_shape(engine, f"ML.{field}")
    matlab = np.array(engine.eval(f"ML.{field}", nargout=1))
    python = np.asarray(loaded[field])
    if int(np.prod(shape)) == 0:
        assert python.size == 0
        assert np.asarray(matlab).size == 0
        return
    assert isinstance(loaded[field], np.ndarray)
    assert python.shape == shape
    np.testing.assert_array_equal(python, matlab)


def _assert_times(engine, loaded) -> None:
    matlab = np.array(engine.eval("ML.times", nargout=1), dtype=np.float64).reshape(-1)
    python = np.asarray(loaded["times"], dtype=np.float64).reshape(-1)
    np.testing.assert_array_equal(python, matlab)


def _matlab_text(engine, expression: str) -> str:
    value = engine.eval(expression, nargout=1)
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "SU":
        return "".join(str(part) for part in array.reshape(-1))
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _matlab_count(engine, field: str) -> int:
    return int(np.array(engine.eval(f"numel(ML.{field})", nargout=1)).reshape(-1)[0])


def _assert_chanlocs(engine, loaded) -> None:
    count = _matlab_count(engine, "chanlocs")
    chanlocs = [] if loaded.get("chanlocs") is None else list(loaded["chanlocs"])
    assert len(chanlocs) == count
    for index, chanloc in enumerate(chanlocs, start=1):
        assert str(chanloc["labels"]) == _matlab_text(engine, f"ML.chanlocs({index}).labels")
        for coord in ("X", "Y", "Z"):
            present = int(np.array(engine.eval(f"isfield(ML.chanlocs, '{coord}')", nargout=1)).reshape(-1)[0])
            if not present:
                continue
            matlab = np.array(engine.eval(f"ML.chanlocs({index}).{coord}", nargout=1), dtype=np.float64).reshape(-1)
            if matlab.size == 0:
                continue
            np.testing.assert_array_equal(np.asarray(chanloc[coord], dtype=np.float64).reshape(-1), matlab)


def _record_text(value) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "SU":
        return "".join(str(part) for part in array.reshape(-1))
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _assert_events(engine, loaded) -> None:
    count = _matlab_count(engine, "event")
    events = [] if loaded.get("event") is None else list(loaded["event"])
    assert len(events) == count
    for index, event in enumerate(events, start=1):
        assert _record_text(event["type"]) == _matlab_text(engine, f"ML.event({index}).type")
        matlab_latency = np.array(engine.eval(f"ML.event({index}).latency", nargout=1), dtype=np.float64).reshape(-1)
        np.testing.assert_array_equal(np.asarray(event["latency"], dtype=np.float64).reshape(-1), matlab_latency)
        matlab_urevent = np.array(engine.eval(f"ML.event({index}).urevent", nargout=1), dtype=np.float64).reshape(-1)
        np.testing.assert_array_equal(np.asarray(event["urevent"], dtype=np.float64).reshape(-1), matlab_urevent - 1)


def _assert_epochs(engine, loaded) -> None:
    count = _matlab_count(engine, "epoch")
    epochs = [] if loaded.get("epoch") is None else list(loaded["epoch"])
    assert len(epochs) == count
    for index, epoch in enumerate(epochs, start=1):
        matlab_events = np.array(engine.eval(f"ML.epoch({index}).event", nargout=1), dtype=np.float64).reshape(-1)
        np.testing.assert_array_equal(np.asarray(epoch["event"], dtype=np.float64).reshape(-1), matlab_events - 1)
        matlab_types = engine.eval(f"ML.epoch({index}).eventtype", nargout=1)
        python_types = epoch.get("eventtype", [])
        if isinstance(python_types, str):
            python_types = [python_types]
        else:
            python_types = [_record_text(item) for item in list(np.asarray(python_types, dtype=object).reshape(-1))]
        if isinstance(matlab_types, str):
            assert python_types == [matlab_types]
        else:
            matlab_list = [_record_text(item) for item in list(np.asarray(matlab_types, dtype=object).reshape(-1))]
            assert python_types == matlab_list


def test_pop_loadset_eeg_variable_keeps_one_by_one_sphere(tmp_path):
    path = tmp_path / "sphere.set"
    data = np.arange(1, 9, dtype=np.float64).reshape(1, 8)
    scipy.io.savemat(
        path,
        {
            "EEG": {
                "setname": "one",
                "filename": path.name,
                "filepath": str(tmp_path),
                "nbchan": [[1.0]],
                "pnts": [[8.0]],
                "trials": [[1.0]],
                "srate": [[128.0]],
                "xmin": [[0.0]],
                "xmax": [[7 / 128]],
                "data": data,
                "icasphere": [[1.0]],
                "icaweights": [[0.5]],
                "icawinv": [[2.0]],
                "icaact": data * 0.5,
                "icachansind": [[1.0]],
                "chanlocs": {"labels": "Cz", "X": [[1.0]], "Y": [[0.0]], "Z": [[0.0]]},
                "event": {"type": "square", "latency": [[4.0]], "urevent": [[1.0]]},
                "saved": "yes",
            }
        },
        appendmat=False,
    )

    loaded = pop_loadset(path)

    assert loaded["data"].shape == (1, 8)
    np.testing.assert_array_equal(loaded["data"], data)
    assert isinstance(loaded["icasphere"], np.ndarray)
    assert loaded["icasphere"].shape == (1, 1)
    np.testing.assert_array_equal(loaded["icasphere"], [[1.0]])
    assert loaded["icaweights"].shape == (1, 1)
    assert loaded["chanlocs"][0]["labels"] == "Cz"
    assert len(loaded["event"]) == 1
    assert loaded["event"][0]["type"] == "square"
    assert loaded["event"][0]["urevent"] == 0


def test_pop_loadset_keeps_cell_struct_and_empty_shapes(tmp_path):
    einfo = np.empty((3, 2), dtype=object)
    for index, value in enumerate(range(6)):
        einfo[np.unravel_index(index, (3, 2), order="F")] = np.array([[float(value)]])
    pinfo = np.empty((2, 5), dtype=object)
    for index, label in enumerate("abcdefghij"):
        pinfo[np.unravel_index(index, (2, 5), order="F")] = label
    structs = np.empty((2, 3), dtype=[("a", "f8"), ("b", "f8")])
    structs["a"] = np.arange(6, dtype=np.float64).reshape(2, 3, order="F")
    structs["b"] = structs["a"] + 10
    row = np.empty((1, 3), dtype=object)
    for index, value in enumerate((1.0, 2.0, 3.0)):
        row[0, index] = np.array([[value]])
    wrapped = np.empty((1, 1), dtype=object)
    wrapped[0, 0] = np.array([[7.0]])
    path = tmp_path / "shapes.set"
    scipy.io.savemat(
        path,
        {
            "EEG": {
                "setname": "shapes",
                "filename": path.name,
                "filepath": str(tmp_path),
                "nbchan": [[1.0]],
                "pnts": [[4.0]],
                "trials": [[1.0]],
                "srate": [[128.0]],
                "xmin": [[0.0]],
                "xmax": [[3 / 128]],
                "data": np.arange(4, dtype=np.float64).reshape(1, 4),
                "chaninfo": {"nodatchans": np.zeros((0, 1))},
                "BIDS": {"eInfo": einfo, "pInfo": pinfo},
                "layout": structs,
                "row": row,
                "wrapped": wrapped,
                "blank": np.zeros((0, 0)),
                "selection": np.zeros((32, 0)),
                "flag": np.array([[True, False], [False, True]]),
            }
        },
        appendmat=False,
    )

    loaded = pop_loadset(path)

    assert loaded["BIDS"]["eInfo"].shape == (3, 2)
    assert [loaded["BIDS"]["eInfo"].reshape(-1, order="F")[index] for index in range(6)] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert loaded["BIDS"]["pInfo"].shape == (2, 5)
    assert loaded["BIDS"]["pInfo"][0, 0] == "a"
    assert loaded["BIDS"]["pInfo"][1, 0] == "b"
    assert loaded["BIDS"]["pInfo"][0, 1] == "c"
    assert loaded["layout"].shape == (2, 3)
    assert loaded["layout"][0, 0]["a"] == 0.0
    assert loaded["layout"][1, 0]["a"] == 1.0
    assert loaded["layout"][0, 1]["a"] == 2.0
    assert loaded["layout"][0, 1]["b"] == 12.0
    assert loaded["row"] == [1.0, 2.0, 3.0]
    assert loaded["wrapped"] == [7.0]
    assert loaded["chaninfo"]["nodatchans"].shape == (0, 1)
    assert loaded["blank"].shape == (0, 0)
    assert loaded["selection"].shape == (32, 0)
    assert loaded["flag"].dtype == np.bool_
    assert loaded["flag"].shape == (2, 2)


def test_pop_loadset_index_fields_and_integral_labels_are_ints(tmp_path):
    path = tmp_path / "indices.set"
    scipy.io.savemat(
        path,
        {
            "EEG": {
                "setname": "indices",
                "filename": path.name,
                "filepath": str(tmp_path),
                "nbchan": [[1.0]],
                "pnts": [[4.0]],
                "trials": [[1.0]],
                "srate": [[128.0]],
                "xmin": [[0.0]],
                "xmax": [[3 / 128]],
                "data": np.arange(4, dtype=np.float64).reshape(1, 4),
                "run": [[1.0]],
                "session": [[2.0]],
                "blank": np.array("", dtype="U"),
                "chanlocs": {"labels": "Cz", "urchan": [[2.0]], "X": [[0.0]], "Y": [[0.0]], "Z": [[0.0]]},
                "event": {"type": "square", "latency": [[4.0]], "urevent": [[3.0]], "epoch": [[1.0]]},
                "urevent": {"type": "square", "latency": [[4.0]]},
            }
        },
        appendmat=False,
    )

    loaded = pop_loadset(path)

    assert loaded["run"] == 1
    assert isinstance(loaded["run"], int)
    assert loaded["session"] == 2
    assert isinstance(loaded["session"], int)
    assert loaded["event"][0]["urevent"] == 2
    assert isinstance(loaded["event"][0]["urevent"], int)
    assert isinstance(loaded["event"][0]["latency"], float)
    assert loaded["chanlocs"][0]["urchan"] == 1
    assert isinstance(loaded["chanlocs"][0]["urchan"], int)
    assert isinstance(loaded["chanlocs"][0]["X"], float)
    assert loaded["blank"].shape == (0, 0)
    assert apply_field_rules("epoch", np.array([[1.0]])) == 1
    assert apply_field_rules("eventurevent", np.array([[4.0]])) == 4
    indices = apply_field_rules("event", np.array([[1.0, 2.0]]))
    assert indices.dtype == int
    assert indices.tolist() == [1, 2]
    assert apply_field_rules("session", np.array([[1.5]])) == 1.5


def test_pop_loadset_single_event_and_single_channel_set(tmp_path):
    eeg = pop_loadset("sample_data/eeglab_data.set")
    eeg = pop_select(eeg, channel=[0])
    eeg["event"] = [dict(eeg["event"][0])]
    out = str(tmp_path / "one_event_one_channel.set")
    pop_saveset(eeg, out)

    reloaded = pop_loadset(out)

    assert len(reloaded["chanlocs"]) == 1
    assert reloaded["chanlocs"][0]["urchan"] == eeg["chanlocs"][0]["urchan"]
    assert len(reloaded["event"]) == 1
    assert reloaded["event"][0]["urevent"] == eeg["event"][0]["urevent"]


def test_pop_loadset_hdf5_fallback_does_not_subtract_icachansind_twice():
    eeg = pop_loadset("sample_data/eeglab_data_epochs_ica_hdf5.set")

    assert np.issubdtype(eeg["icachansind"].dtype, np.integer)
    assert eeg["icachansind"][0] == 0


def test_pop_loadset_corrupt_v7_set_surfaces_real_error_not_h5py(tmp_path):
    # A corrupt non-HDF5 .set must surface the real scipy parse error, not be silently
    # rerouted to the HDF5 loader where h5py raises a cryptic "file signature not found".
    corrupt = tmp_path / "corrupt.set"
    corrupt.write_bytes(b"MATLAB 5.0 MAT-file, corrupt" + b"\x00" * 100 + b"\xff" * 50)

    with pytest.raises(Exception) as excinfo:
        pop_loadset(str(corrupt))

    message = str(excinfo.value).lower()
    assert "file signature not found" not in message
    assert "mat file" in message or "unknown mat file" in message


def test_pop_loadset_h5_unicode_decodes_via_general_path(tmp_path):
    # The general uint16 -> UTF-8 decode (no hard-coded emoji branch) returns the
    # correct character. The bytes below are UTF-8 for U+1F496 (sparkling heart).
    filepath = tmp_path / "unicode.set"
    with h5py.File(filepath, "w") as f:
        eeg_group = f.create_group("EEG")
        unicode_bytes = np.array([104, 101, 108, 108, 111, 32, 240, 159, 146, 150], dtype=np.uint16)
        eeg_group.create_dataset("unicode_string", data=unicode_bytes)
        eeg_group.create_dataset("srate", data=np.array([[500.0]]))
        eeg_group.create_dataset("nbchan", data=np.array([[4]]))
        eeg_group.create_dataset("pnts", data=np.array([[100]]))
        eeg_group.create_dataset("trials", data=np.array([[1]]))
        eeg_group.create_dataset("xmin", data=np.array([[-1.0]]))
        eeg_group.create_dataset("xmax", data=np.array([[1.0]]))
        eeg_group.create_dataset("data", data=np.zeros((4, 100), dtype=np.float32))

    eeg = pop_loadset_h5(str(filepath))

    assert eeg["unicode_string"] == "hello \U0001f496"

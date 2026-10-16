import h5py
import numpy as np
import pytest
import scipy.io

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
    raise AssertionError(kind)


@pytest.mark.parametrize("savemode", ["onefile", "twofiles"])
@pytest.mark.parametrize("kind", ["one_channel", "one_channel_epochs", "one_sample", "one_component"])
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
    """,
}


@pytest.mark.parametrize("savemode", ["onefile", "twofiles"])
@pytest.mark.parametrize("kind", ["one_channel", "one_channel_epochs", "one_sample", "one_component"])
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
    if kind != "one_component":
        return
    for field in ("icaweights", "icasphere", "icawinv"):
        assert loaded[field].shape == _matlab_shape(eeglab_matlab_engine, f"ML.{field}")


def test_pop_loadset_single_event_and_single_channel_set(tmp_path):
    # A 1x1 MATLAB struct array is read by scipy as a bare dict, not a list.
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

import h5py
import numpy as np
import pytest
import scipy.io

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


def test_pop_loadset_single_event_and_single_channel_set(tmp_path):
    # A 1x1 MATLAB struct array is read by scipy as a bare dict, not a list.
    eeg = pop_loadset("sample_data/eeglab_data.set")
    eeg = pop_select(eeg, channel=[0])
    eeg["event"] = [dict(eeg["event"][0])]
    out = str(tmp_path / "one_event_one_channel.set")
    pop_saveset(eeg, out)

    reloaded = pop_loadset(out)

    assert len(reloaded["chanlocs"]) == 1
    assert reloaded["data"].shape == (1, eeg["data"].shape[1])
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


def test_pop_loadset_h5_unicode_decodes_uint16_code_units(tmp_path):
    # U+1F496 is the UTF-16 surrogate pair U+D83D U+DC96, stored as a MATLAB char row.
    filepath = tmp_path / "unicode.set"
    codes = np.array([[ord(c)] for c in "hello "] + [[0xD83D], [0xDC96]], dtype=np.uint16)
    with h5py.File(filepath, "w") as handle:
        eeg_group = handle.create_group("EEG")
        text = eeg_group.create_dataset("unicode_string", data=codes)
        text.attrs["MATLAB_class"] = "char"
        text.attrs["MATLAB_int_decode"] = np.int32(2)
        for name, value in (
            ("srate", 500.0),
            ("nbchan", 4),
            ("pnts", 100),
            ("trials", 1),
            ("xmin", -1.0),
            ("xmax", 1.0),
        ):
            field = eeg_group.create_dataset(name, data=np.array([[value]], dtype=np.float64))
            field.attrs["MATLAB_class"] = "double"
        data = eeg_group.create_dataset("data", data=np.zeros((100, 4), dtype=np.float32))
        data.attrs["MATLAB_class"] = "single"

    eeg = pop_loadset_h5(str(filepath))

    assert eeg["unicode_string"] == "hello \U0001f496"
    assert eeg["data"].shape == (4, 100)

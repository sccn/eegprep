import h5py
import numpy as np
import pytest

from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_loadset_h5 import pop_loadset_h5


def test_pop_loadset_marks_loaded_dataset_justloaded():
    eeg = pop_loadset("sample_data/eeglab_data.set")

    assert eeg["saved"] == "justloaded"


def test_pop_loadset_does_not_print_loaded_path(capsys):
    pop_loadset("sample_data/eeglab_data.set")

    assert capsys.readouterr().out == ""


def test_pop_loadset_normalizes_empty_icachansind_to_integer_array(caplog):
    eeg = pop_loadset("sample_data/eeglab_data.set")

    assert np.issubdtype(eeg["icachansind"].dtype, np.integer)
    assert eeg["icachansind"].shape == (0,)
    assert "Field 'icachansind' is expected to be a numpy array of integers" not in caplog.text


def test_pop_loadset_normalizes_nonempty_icachansind_to_zero_based_integers():
    eeg = pop_loadset("sample_data/eeglab_data_epochs_ica.set")

    assert np.issubdtype(eeg["icachansind"].dtype, np.integer)
    np.testing.assert_array_equal(eeg["icachansind"][:5], np.array([0, 1, 2, 3, 4]))


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


def test_pop_loadset_routes_hdf5_file_to_h5_loader(tmp_path):
    # A genuine HDF5 .set is detected by its signature and loaded by the HDF5 path.
    filepath = tmp_path / "hdf5.set"
    with h5py.File(filepath, "w") as f:
        eeg_group = f.create_group("EEG")
        eeg_group.create_dataset("srate", data=np.array([[500.0]]))
        eeg_group.create_dataset("nbchan", data=np.array([[4]]))
        eeg_group.create_dataset("pnts", data=np.array([[100]]))
        eeg_group.create_dataset("trials", data=np.array([[1]]))
        eeg_group.create_dataset("xmin", data=np.array([[-1.0]]))
        eeg_group.create_dataset("xmax", data=np.array([[1.0]]))
        eeg_group.create_dataset("data", data=np.zeros((4, 100), dtype=np.float32))

    eeg = pop_loadset(str(filepath))

    assert eeg["nbchan"] == 4
    assert eeg["data"].shape == (4, 100)


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

import numpy as np

from eegprep.functions.popfunc.pop_loadset import pop_loadset


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

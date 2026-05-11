from eegprep.functions.popfunc.pop_loadset import pop_loadset


def test_pop_loadset_marks_loaded_dataset_justloaded():
    eeg = pop_loadset("sample_data/eeglab_data.set")

    assert eeg["saved"] == "justloaded"


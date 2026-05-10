from eegprep.functions.popfunc.eeg_emptyset import eeg_emptyset


def test_eeg_emptyset_includes_eeglab_reference_defaults():
    eeg = eeg_emptyset()

    assert eeg["ref"] == "common"
    assert eeg["urchanlocs"] == []
    assert eeg["run"] == []
    assert eeg["eventdescription"] == []
    assert eeg["epochdescription"] == []
    assert eeg["datfile"] == ""
    assert eeg["roi"] == {}

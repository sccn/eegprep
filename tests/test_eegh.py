from eegprep.functions.adminfunc.eegh import eegh


def test_eegh_records_only_non_empty_commands():
    history = []

    assert eegh(" EEG = pop_reref(EEG); ", history) == "EEG = pop_reref(EEG);"
    assert eegh("", history) == ""
    assert eegh(None, history) == ""
    assert history == ["EEG = pop_reref(EEG);"]

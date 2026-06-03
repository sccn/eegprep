from eegprep.functions.adminfunc.eegh import eegh


def test_eegh_records_only_non_empty_commands():
    history = []

    assert eegh(" EEG = pop_reref(EEG); ", history) == "EEG = pop_reref(EEG);"
    assert eegh("", history) == ""
    assert eegh(None, history) == "1. EEG = pop_reref(EEG);"
    assert history == ["EEG = pop_reref(EEG);"]


def test_eegh_displays_finds_removes_and_clears_newest_first():
    history = []
    eegh("EEG = pop_loadset('sample.set');", history)
    eegh("EEG = pop_resample(EEG, 64);", history)
    eegh("EEG = pop_reref(EEG, []);", history)

    assert eegh(None, history).splitlines() == [
        "1. EEG = pop_reref(EEG, []);",
        "2. EEG = pop_resample(EEG, 64);",
        "3. EEG = pop_loadset('sample.set');",
    ]
    assert eegh(2, history) == "EEG = pop_resample(EEG, 64);"

    assert eegh(-1, history) == ""
    assert history == ["EEG = pop_loadset('sample.set');", "EEG = pop_resample(EEG, 64);"]

    assert eegh(0, history) == ""
    assert history == []


def test_eegh_command_appends_to_eeg_history():
    eeg = {"history": "EEG = pop_loadset('sample.set');"}

    assert eegh("EEG = pop_resample(EEG, 64);", eeg) == "EEG = pop_resample(EEG, 64);"

    assert eeg["history"].splitlines() == [
        "EEG = pop_loadset('sample.set');",
        "EEG = pop_resample(EEG, 64);",
    ]

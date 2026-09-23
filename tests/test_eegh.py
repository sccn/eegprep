import pytest

from eegprep.functions.adminfunc.eegh import eegh, eegh_find
from tests.eeglab_tests import eeglab_test


@eeglab_test("unittesting_adminfunc/eegh/pass_new_command.m", "test_pass_new_command")
@eeglab_test("unittesting_adminfunc/eegh/pass_empty_command.m", "test_pass_empty_command")
def test_eegh_records_only_non_empty_commands():
    history = []

    assert eegh(" EEG = pop_reref(EEG); ", history) == "EEG = pop_reref(EEG);"
    assert eegh("", history) == ""
    assert eegh(None, history) == "1. EEG = pop_reref(EEG);"
    assert history == ["EEG = pop_reref(EEG);"]


@eeglab_test("unittesting_adminfunc/eegh/pass_return_commands.m", "test_pass_return_commands")
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


@eeglab_test("unittesting_adminfunc/eegh/pass_add_history.m", "test_pass_add_history")
def test_eegh_command_appends_to_eeg_history():
    eeg = {"history": "EEG = pop_loadset('sample.set');"}

    assert eegh("EEG = pop_resample(EEG, 64);", eeg) == "EEG = pop_resample(EEG, 64);"

    assert eeg["history"].splitlines() == [
        "EEG = pop_loadset('sample.set');",
        "EEG = pop_resample(EEG, 64);",
    ]


@eeglab_test("unittesting_adminfunc/eeg_hist/pass_general.m", "test_pass_general")
def test_eegh_initializes_empty_dataset_history_with_the_command():
    eeg = {}

    eegh("a command", eeg)

    assert eeg["history"] == "a command;"


def test_eegh_eeg_history_dedup_compares_last_line_exactly():
    eeg = {"history": "AEEG = pop_reref(EEG);"}

    eegh("EEG = pop_reref(EEG);", eeg)
    eegh("EEG = pop_reref(EEG);", eeg)

    assert eeg["history"].splitlines() == ["AEEG = pop_reref(EEG);", "EEG = pop_reref(EEG);"]


@eeglab_test("unittesting_adminfunc/eegh/pass_insert_command.m", "test_pass_insert_command")
def test_eegh_inserts_a_new_command_after_existing_history():
    history = ["command1", "command2"]

    eegh("command3", history)

    assert history == ["command1", "command2", "command3"]


@eeglab_test("unittesting_adminfunc/eegh/pass_find_command.m", "test_pass_find_command")
@eeglab_test("unittesting_adminfunc/eegh/pass_find_command2.m", "test_pass_find_command2")
def test_eegh_find_returns_most_recent_match_or_empty():
    history = ["command1", "command2", "command3", "comm4"]

    assert eegh_find(history, "comma") == "command3"
    assert eegh_find(history, "command4") == ""


@eeglab_test("unittesting_adminfunc/eegh/pass_unstack_command.m", "test_pass_unstack_command")
def test_eegh_removes_the_requested_number_of_recent_commands():
    history = ["command1", "command2", "command3"]

    assert eegh(-2, history) == ""
    assert history == ["command1"]


@eeglab_test("unittesting_adminfunc/eegh/pass_delete_commands.m", "test_pass_delete_commands")
def test_eegh_zero_clears_command_history():
    history = ["command1", "command2"]

    assert eegh(0, history) == ""
    assert history == []


@eeglab_test("unittesting_adminfunc/eegh/fail_get_empty.m", "test_fail_get_empty")
def test_eegh_selecting_from_empty_history_returns_empty():
    assert eegh(1, []) == ""


@eeglab_test("unittesting_adminfunc/eegh/pass_add_multiple_history.m", "test_pass_add_multiple_history")
@eeglab_test("unittesting_adminfunc/eegh/pass_add_multiple_history2.m", "test_pass_add_multiple_history2")
@pytest.mark.parametrize("include_history", [False, True])
def test_eegh_marks_commands_applied_to_multiple_datasets(include_history):
    datasets = [{"data": [1]}, {"data": [2]}]
    if include_history:
        for eeg in datasets:
            eeg["history"] = ""

    assert eegh("command3", datasets) == "command3"

    assert [eeg["data"] for eeg in datasets] == [[1], [2]]
    assert [eeg["history"] for eeg in datasets] == [
        "% multiple datasets command: command3;",
        "% multiple datasets command: command3;",
    ]

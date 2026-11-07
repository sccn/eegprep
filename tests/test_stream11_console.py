"""Stream 2 handover: translate MATLAB colon vectors only for console input."""

import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace, _console_python_command
from eegprep.functions.guifunc.select_multiple_datasets import select_multiple_datasets
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_copyset import pop_copyset
from tests.test_gui_main_window import _demo_eeg


@pytest.mark.parametrize(
    "matlab, expected",
    [
        ("[1:32]", list(range(1, 33))),
        ("[1:2:7]", [1, 3, 5, 7]),
        ("[5:-2:1]", [5, 3, 1]),
        ("[5:3]", []),
        ("[-3:-1]", [-3, -2, -1]),
        ("[1:3 8 10:2:14]", [1, 2, 3, 8, 10, 12, 14]),
        ("[1:3;4:6]", [[1, 2, 3], [4, 5, 6]]),
        ("[0:0.25:1]", [0, 0.25, 0.5, 0.75, 1]),
    ],
)
def test_colon_ranges_become_executable_python_without_changing_history(matlab, expected):
    source = f"indices = {matlab};"
    session = EEGPrepSession()
    session.add_history(source)
    translated = _console_python_command(session.LASTCOM)
    namespace = {}
    exec(translated, namespace)
    assert namespace["indices"] == expected
    assert session.ALLCOM == [source]
    assert session.LASTCOM == source


def test_long_contiguous_range_keeps_console_command_short():
    translated = _console_python_command("indices = [1:1000000];")
    assert len(translated) < 100
    assert "range(" in translated


def test_colon_inside_matlab_string_stays_text():
    translated = _console_python_command("label = 'channels [1:32]';")
    namespace = {}
    exec(translated, namespace)
    assert namespace["label"] == "channels [1:32]"


def test_channel_colon_range_still_translates_matlab_indices():
    source = "EEG = pop_select(EEG, 'channel', [1:32]);"
    translated = _console_python_command(source)
    namespace = {"EEG": None, "pop_select": lambda _eeg, **options: options["channel"]}
    exec(translated, namespace)
    assert namespace["EEG"] == list(range(32))


def test_f94_generated_selection_and_copy_commands_replay_in_console():
    session = EEGPrepSession()
    for name in ("first", "second"):
        session.store_current(dict(_demo_eeg(), setname=name), new=True)
    workspace = EEGPrepConsoleWorkspace(session)
    try:
        _, selection_command = select_multiple_datasets(session, [1, 2], return_com=True)
        session.retrieve(2)
        session.add_history(selection_command)
        workspace.execute_history_command(selection_command)
        assert session.CURRENTSET == [1, 2]
        session.retrieve(1)
        _, _, _, copy_command = pop_copyset(session.ALLEEG, 1, 3, return_com=True)
        session.add_history(copy_command)
        workspace.execute_history_command(copy_command)
        assert session.CURRENTSET == [3]
        assert session.EEG["setname"] == "first"
    finally:
        workspace.close()

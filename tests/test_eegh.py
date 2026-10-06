import numpy as np
import pytest

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.adminfunc.eegh import eegh
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests.assertions import assert_matlab_struct_near


_EEGH_WRAPPER = "unittesting_adminfunc/eegh/adminfunc_eegh_wrapperTest.m"


@pytest.fixture
def reference_history(eeglab_backend, request):
    empty = np.empty((0, 0))
    if request.config.getoption("--eeglab-backend") == "matlab":
        saved = eeglab_backend("eegprep_test_history_state", nargout=2)

        def set_history(commands):
            eeglab_backend("eegprep_test_history_state", commands, empty, nargout=0)

        def call_eegh(*args, nargout=1):
            return eeglab_backend("eegh", *args, nargout=nargout)

        def read_history():
            return eeglab_backend("eegprep_test_history_state")

        set_history(empty)
        try:
            yield set_history, call_eegh, read_history
        finally:
            eeglab_backend("eegprep_test_history_state", *saved, nargout=0)
        return

    session = EEGPrepSession()
    workspace = EEGPrepConsoleWorkspace(session)

    def set_history(commands):
        # The real Python session stores oldest first; source ALLCOM cells are
        # newest first. Map storage orientation only, leaving every action intact.
        session.apply_workspace_state(allcom=list(commands.ravel(order="F"))[::-1], lastcom="")

    def call_eegh(*args, nargout=1):
        result = workspace.namespace["eegh"](*args)
        return result if nargout else None

    def read_history():
        return np.array([session.ALLCOM[::-1]], dtype=object)

    set_history(empty)
    try:
        yield set_history, call_eegh, read_history
    finally:
        workspace.close()


@eeglab_test(_EEGH_WRAPPER, "test_pass_new_command")
def test_reference_eegh_new_command(eeglab_backend, reference_history):
    _, call, history = reference_history
    call("command1", nargout=0)
    assert_matlab_struct_near(history(), np.array([["command1"]], dtype=object))


@eeglab_test(_EEGH_WRAPPER, "test_pass_empty_command")
def test_reference_eegh_empty_command(eeglab_backend, reference_history):
    _, call, _ = reference_history
    call(np.empty((0, 0)), nargout=0)


@eeglab_test(_EEGH_WRAPPER, "test_pass_insert_command")
def test_reference_eegh_insert_command(eeglab_backend, reference_history):
    setup, call, history = reference_history
    setup(np.array([["command2", "command1"]], dtype=object))
    call("command3", nargout=0)
    assert_matlab_struct_near(history(), np.array([["command3", "command2", "command1"]], dtype=object))


@eeglab_test(_EEGH_WRAPPER, "test_pass_execute_command")
def test_reference_eegh_execute_command(eeglab_backend, reference_history):
    setup, call, history = reference_history
    setup(np.array([["disp(3);", "disp(2);", "disp(1);"]], dtype=object))
    call(2.0, nargout=0)
    assert_matlab_struct_near(history(), np.array([["disp(2);", "disp(3);", "disp(2);", "disp(1);"]], dtype=object))


@eeglab_test(_EEGH_WRAPPER, "test_pass_find_command")
def test_reference_eegh_find_command(eeglab_backend, reference_history):
    setup, call, _ = reference_history
    setup(np.array([["comm4", "command3", "command2", "command1"]], dtype=object))
    assert_matlab_struct_near(call("find", "comma"), "command3")


@eeglab_test(_EEGH_WRAPPER, "test_pass_find_command2")
def test_reference_eegh_find_absent_command(eeglab_backend, reference_history):
    setup, call, _ = reference_history
    setup(np.array([["comm4", "command3", "command2", "command1"]], dtype=object))
    result = call("find", "command4")
    assert not result if isinstance(result, str) else np.asarray(result).size == 0


@eeglab_test(_EEGH_WRAPPER, "test_pass_unstack_command")
def test_reference_eegh_unstack_command(eeglab_backend, reference_history):
    setup, call, history = reference_history
    setup(np.array([["command3", "command2", "command1"]], dtype=object))
    call(-1.0, nargout=0)
    assert_matlab_struct_near(history(), np.array([["command2", "command1"]], dtype=object))


@eeglab_test(_EEGH_WRAPPER, "test_pass_delete_commands")
def test_reference_eegh_delete_commands(eeglab_backend, reference_history):
    setup, call, history = reference_history
    setup(np.array([["command2", "command1"]], dtype=object))
    call(0.0, nargout=0)
    assert np.asarray(history()).size == 0


@eeglab_test(_EEGH_WRAPPER, "test_pass_add_history")
def test_reference_eegh_add_history(eeglab_backend, reference_history):
    setup, call, history = reference_history
    eeg = eeglab_backend("eeg_emptyset")
    setup(np.array([["command2", "command1"]], dtype=object))
    result = call("command3", eeg)
    assert_matlab_struct_near(history(), np.array([["command3", "command2", "command1"]], dtype=object))
    assert_matlab_struct_near(result["history"], "\ncommand3")


def _reference_multiple_eeg(eeglab_backend):
    first = eeglab_backend("eeg_emptyset")
    first["data"] = np.array([[1.0]])
    second = eeglab_backend("eeg_emptyset")
    second["data"] = np.array([[2.0]])
    return np.array([[tuple(first.values()), tuple(second.values())]], dtype=[(field, object) for field in first])


def _reference_multiple_history(reference_history, eeg):
    setup, call, history = reference_history
    setup(np.array([["command2", "command1"]], dtype=object))
    result = call("command3", eeg)
    assert_matlab_struct_near(history(), np.array([["command3", "command2", "command1"]], dtype=object))
    for index in range(2):
        assert_matlab_struct_near(result["history"].ravel(order="F")[index], "\n% multiple datasets command: command3")
        assert_matlab_struct_near(result["data"].ravel(order="F")[index], np.array([[float(index + 1)]]))


@eeglab_test(_EEGH_WRAPPER, "test_pass_add_multiple_history")
def test_reference_eegh_add_multiple_history(eeglab_backend, reference_history):
    eeg = _reference_multiple_eeg(eeglab_backend)
    eeg["history"][0, 0] = np.empty((0, 0))
    eeg["history"][0, 1] = np.empty((0, 0))
    _reference_multiple_history(reference_history, eeg)


@eeglab_test(_EEGH_WRAPPER, "test_pass_add_multiple_history2")
def test_reference_eegh_add_multiple_history_default_empty(eeglab_backend, reference_history):
    _reference_multiple_history(reference_history, _reference_multiple_eeg(eeglab_backend))


@eeglab_test("unittesting_adminfunc/eeg_hist/adminfunc_eeg_hist_wrapperTest.m", "test_pass_general")
def test_reference_eeg_hist_general(eeglab_backend):
    eeg = eeglab_backend("eeg_emptyset")
    result = eeglab_backend("eeg_hist", eeg, "a command")
    assert_matlab_struct_near(result["history"], "\na command")


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


def test_eegh_eeg_history_dedup_compares_last_line_exactly():
    eeg = {"history": "AEEG = pop_reref(EEG);"}

    eegh("EEG = pop_reref(EEG);", eeg)
    eegh("EEG = pop_reref(EEG);", eeg)

    assert eeg["history"].splitlines() == ["AEEG = pop_reref(EEG);", "EEG = pop_reref(EEG);"]


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

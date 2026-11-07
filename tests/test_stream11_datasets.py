"""Dataset workflow and MATLAB history regressions from issue #556."""

from copy import deepcopy

import numpy as np
import pytest

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QDialogButtonBox, QFileDialog, QLineEdit, QMessageBox, QPushButton
except ImportError:
    pytest.skip("PySide6 is required for GUI workflows", allow_module_level=True)

from eegprep.functions.adminfunc.console import EEGPrepConsoleWorkspace
from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.select_multiple_datasets import select_multiple_datasets
from eegprep.functions.guifunc.session import EEGPrepSession
from eegprep.functions.popfunc.pop_copyset import pop_copyset
from eegprep.functions.popfunc.pop_newset import pop_newset
from tests.test_gui_main_window import _demo_eeg
from tests.test_stream11_menus import qapp as qapp

pytestmark = [pytest.mark.gui, pytest.mark.modal_dialogs]


@pytest.fixture
def dialogs(qapp):
    pending, seen, failures = [], [], []

    def respond():
        dialog = qapp.activeModalWidget()
        if dialog is None:
            return
        seen.append(dialog.windowTitle())
        if not pending:
            failures.append(f"Unexpected dialog: {dialog.windowTitle()}")
            dialog.reject()
            return
        handler = pending.pop(0)
        try:
            handler(dialog)
        except Exception as exc:
            failures.append(str(exc))
            dialog.reject()

    timer = QTimer()
    timer.timeout.connect(respond)
    timer.start(10)
    yield pending, seen
    timer.stop()
    assert not failures, failures
    assert not pending, "A required dialog did not open"


def accept(dialog):
    button = dialog.findChild(QPushButton, "ok")
    if button is None:
        button = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    button.click()


def make_session():
    session = EEGPrepSession()
    for name in ("first", "second"):
        eeg = _demo_eeg(ica=False)
        eeg["setname"] = name
        session.store_current(eeg, new=True)
    return session


def test_f27_newset_history_has_concrete_old_index_and_no_dialog_on_replay():
    alleeg, eeg, index, command = pop_newset([], _demo_eeg(), 0, setname="O'Brien")
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'setname', 'O''Brien', 'gui', 'off');"
    _, _, _, command = pop_newset(alleeg, eeg, index, overwrite="on")
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'overwrite', 'on', 'gui', 'off');"


def test_f49_copy_history_has_three_outputs():
    _, _, index, command = pop_copyset([_demo_eeg()], 1, 2, return_com=True)
    assert index == 2
    assert command == "[ALLEEG EEG CURRENTSET] = pop_copyset( ALLEEG, 1, 2);"


def test_f94_selection_and_processing_history_use_original_selection():
    session = make_session()
    _, command = select_multiple_datasets(session, [1, 2], return_com=True)
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2, 'retrieve', [1 2], 'study', 0);"
    _, _, indices, command = pop_newset(session.ALLEEG, session.EEG, [1, 2], overwrite="on", study=0)
    assert indices == [1, 2]
    assert command == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, [1 2], 'study', 0);"


@pytest.mark.parametrize("proceed", [False, True])
def test_f90_warning_precedes_multi_processing_and_cancel_preserves_state(qapp, dialogs, proceed):
    session = make_session()
    session.retrieve([1, 2])
    window = build_main_window(session, include_plugins=False)
    pending, seen = dialogs
    before = deepcopy(session.ALLEEG)

    def warning(dialog):
        assert isinstance(dialog, QMessageBox)
        assert dialog.windowTitle() == "Multiple dataset warning"
        assert "overwritten" in dialog.text()
        button = next(button for button in dialog.buttons() if button.text() == ("Proceed" if proceed else "Cancel"))
        button.click()

    pending.append(warning)
    if proceed:

        def resample(dialog):
            dialog.findChild(QLineEdit, "freq").setText("50")
            accept(dialog)

        pending.append(resample)
    try:
        window.dispatcher.dispatch("pop_resample", window.window)
        assert seen[0] == "Multiple dataset warning"
        assert session.CURRENTSET == [1, 2]
        if proceed:
            assert [eeg["srate"] for eeg in session.EEG] == [50, 50]
            assert len(session.ALLEEG) == 2
            assert session.ALLCOM[-1] == "[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, [1 2], 'study', 0);"
        else:
            assert not session.ALLCOM
            for old, current in zip(before, session.ALLEEG):
                np.testing.assert_array_equal(old["data"], current["data"])
                assert old["srate"] == current["srate"]
    finally:
        window.window.close()


@pytest.mark.parametrize("cancel_newset", [False, True])
def test_f42_merge_opens_newset_and_cancel_preserves_datasets(qapp, dialogs, cancel_newset):
    session = make_session()
    window = build_main_window(session, include_plugins=False)
    pending, seen = dialogs
    pending.append(accept)

    def newset(dialog):
        assert "pop_newset" in dialog.windowTitle()
        if cancel_newset:
            dialog.reject()
        else:
            dialog.findChild(QLineEdit, "setname").setText("Combined")
            accept(dialog)

    pending.append(newset)
    try:
        window.dispatcher.dispatch("pop_mergeset", window.window)
        assert len(seen) == 2
        if cancel_newset:
            assert len(session.ALLEEG) == 2
            assert not session.ALLCOM
        else:
            assert session.CURRENTSET == [3]
            assert session.EEG["setname"] == "Combined"
            assert session.EEG["pnts"] == 80
            assert "pop_mergeset" in session.ALLCOM[-2]
            assert "pop_newset(ALLEEG, EEG, 2," in session.ALLCOM[-1]
    finally:
        window.window.close()


def test_f38_empty_result_never_replaces_current_dataset(qapp):
    session = make_session()
    window = build_main_window(session, include_plugins=False)
    empty = _demo_eeg()
    empty.update(data=np.zeros((2, 0)), pnts=0)
    try:
        with pytest.raises(ValueError, match="pop_select.*data.*selection"):
            window.dispatcher._commit_processed_dataset_from_gui(empty, command="EEG = pop_select(EEG);", parent=None)
        assert session.CURRENTSET == [2]
        assert session.EEG["pnts"] == 40
        assert not session.ALLCOM
    finally:
        window.window.close()


def test_f27_exported_session_history_is_matlab_script(qapp, dialogs, tmp_path):
    session = make_session()
    session.add_history("EEG = pop_loadset('demo.set');")
    window = build_main_window(session, include_plugins=False, native_file_dialogs=False)
    path = tmp_path / "session.m"
    pending, _ = dialogs

    def save(dialog):
        assert isinstance(dialog, QFileDialog)
        dialog.findChild(QLineEdit, "fileNameEdit").setText(str(path))
        dialog.accept()

    pending.append(save)
    try:
        window.dispatcher.dispatch("pop_saveh:session", window.window)
        text = path.read_text()
        assert "[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;" in text
        assert "eeglab redraw;" in text
        assert "eegprep.eeglab" not in text
    finally:
        window.window.close()


def test_f27_new_import_records_store_command(qapp):
    session = EEGPrepSession()
    window = build_main_window(session, include_plugins=False)
    try:
        window.dispatcher._store_current_from_gui(_demo_eeg(), new=True, command="EEG = pop_loadset('demo.set');")
        assert session.ALLCOM == [
            "EEG = pop_loadset('demo.set');",
            "[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG, 0);",
        ]
    finally:
        window.window.close()


def test_f27_save_history_separates_filename_and_filepath(qapp, dialogs, tmp_path):
    session = make_session()
    window = build_main_window(session, include_plugins=False, native_file_dialogs=False)
    pending, _ = dialogs
    path = tmp_path / "saved.set"

    def save(dialog):
        dialog.findChild(QLineEdit, "fileNameEdit").setText(str(path))
        dialog.accept()

    pending.append(save)
    try:
        window.dispatcher.dispatch("pop_saveset", window.window)
        assert path.is_file()
        assert session.LASTCOM == f"EEG = pop_saveset(EEG, 'filename', 'saved.set', 'filepath', '{tmp_path}');"
    finally:
        window.window.close()


@pytest.mark.parametrize("cancel", [True, False])
def test_f29_clear_keeps_s1_dataset_number_dialog(qapp, dialogs, cancel):
    session = make_session()
    window = build_main_window(session, include_plugins=False)
    pending, _ = dialogs

    def delete(dialog):
        assert "pop_delset" in dialog.windowTitle()
        assert dialog.findChild(QLineEdit).text() == "2"
        if cancel:
            dialog.reject()
        else:
            accept(dialog)

    pending.append(delete)
    try:
        window.dispatcher.dispatch("pop_delset", window.window)
        assert len(session.ALLEEG) == (2 if cancel else 1)
        assert session.CURRENTSET == ([2] if cancel else [1])
    finally:
        window.window.close()


def test_f28_python_script_updates_shared_gui_and_console(qapp, dialogs, tmp_path):
    session = make_session()
    window = build_main_window(session, include_plugins=False, native_file_dialogs=False)
    workspace = EEGPrepConsoleWorkspace(session, window=window)
    path = tmp_path / "rename.py"
    path.write_text("EEG = dict(EEG, setname='scripted')\n")
    pending, _ = dialogs

    def choose(dialog):
        dialog.findChild(QLineEdit, "fileNameEdit").setText(str(path))
        dialog.accept()

    pending.append(choose)
    try:
        window.dispatcher.dispatch("pop_runscript", window.window)
        assert workspace.namespace["EEG"]["setname"] == "scripted"
        assert "scripted" in window.heading.text()
        assert session.ALLEEG[1]["setname"] == "scripted"
    finally:
        workspace.close()
        window.window.close()


@pytest.mark.parametrize("discard", [False, True])
def test_quit_requires_explicit_choice_for_unsaved_datasets(qapp, dialogs, discard):
    session = make_session()
    window = build_main_window(session, include_plugins=False)
    window.show()
    pending, _ = dialogs

    def choose(dialog):
        assert isinstance(dialog, QMessageBox)
        assert "unsaved" in dialog.text().lower()
        button = dialog.button(QMessageBox.Discard if discard else QMessageBox.Cancel)
        button.click()

    pending.append(choose)
    try:
        window.dispatcher.dispatch("quit", window.window)
        assert window.window.isVisible() is not discard
        assert len(session.ALLEEG) == 2
    finally:
        window.window.close()

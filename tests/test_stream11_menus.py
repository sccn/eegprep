"""Regression coverage for Stream 11 menu lifetime and summary findings."""

import gc

import pytest
import numpy as np

try:
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit, QMessageBox
except ImportError:
    pytest.skip("PySide6 is required for GUI workflows", allow_module_level=True)

from eegprep.functions.guifunc.main_window import _reference_state, _size_mb, _summary_for_session

from eegprep.functions.guifunc.main_window import build_main_window
from eegprep.functions.guifunc.session import EEGPrepSession
from tests.test_gui_main_window import _demo_eeg, _find_menu_action

pytestmark = pytest.mark.gui


def test_f54_f121_f30_refresh_keeps_open_menu_objects(qapp):
    session = EEGPrepSession()
    session.store_current(_demo_eeg())
    window = build_main_window(session, include_plugins=False)
    window.show()
    before = window.window.menuBar().actions()
    tools = window.open_menu("Tools")
    resample = _find_menu_action(tools, "pop_resample")
    try:
        for _ in range(3):
            window.refresh()
            gc.collect()
            qapp.processEvents()
        assert window.window.menuBar().actions() == before
        assert _find_menu_action(tools, "pop_resample") is resample
        assert resample.isEnabled()
    finally:
        window.window.close()


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.mark.modal_dialogs
def test_f121_failed_import_keeps_file_menu(qapp, tmp_path):
    corrupt = tmp_path / "corrupt.set"
    corrupt.write_text("not a MAT file")
    window = build_main_window(include_plugins=False, native_file_dialogs=False)
    window.show()
    before = window.window.menuBar().actions()
    seen = []

    def respond():
        dialog = qapp.activeModalWidget()
        if isinstance(dialog, QFileDialog):
            seen.append("file")
            dialog.findChild(QLineEdit, "fileNameEdit").setText(str(corrupt))
            dialog.accept()
        elif isinstance(dialog, QMessageBox):
            seen.append("error")
            gc.collect()
            dialog.accept()

    timer = QTimer()
    timer.timeout.connect(respond)
    timer.start(10)
    try:
        window.open_menu("File")
        _find_menu_action(window.window.menuBar(), "pop_loadset").trigger()
        window.refresh()
        qapp.processEvents()
        assert seen == ["file", "error"]
        assert window.window.menuBar().actions() == before
        assert _find_menu_action(window.window.menuBar(), "pop_loadset").isEnabled()
    finally:
        timer.stop()
        window.window.close()


def test_f1_f95_single_dataset_shows_multi_selection(qapp):
    session = EEGPrepSession()
    session.store_current(_demo_eeg())
    window = build_main_window(session, include_plugins=False)
    try:
        action = _find_menu_action(window.window.menuBar(), "select_multiple_datasets")
        assert action is not None and action.isEnabled()
    finally:
        window.window.close()


def test_f38_empty_current_dataset_still_allows_retrieval(qapp):
    session = EEGPrepSession()
    session.store_current(_demo_eeg())
    empty = _demo_eeg()
    empty.update(data=np.zeros((2, 0)), pnts=0)
    session.apply_workspace_state(alleeg=[session.EEG, empty], eeg=empty, currentset=2)
    window = build_main_window(session, include_plugins=False)
    try:
        datasets = next(action for action in window.window.menuBar().actions() if action.text() == "Datasets")
        assert datasets.isEnabled()
        _find_menu_action(datasets.menu(), "retrieve_dataset:1").trigger()
        assert session.CURRENTSET == [1]
        assert session.EEG["pnts"] == 40
    finally:
        window.window.close()


def test_f5_no_ica_keeps_component_scroll_and_iclabel_menu_enabled(qapp):
    session = EEGPrepSession()
    session.store_current(_demo_eeg(ica=False))
    window = build_main_window(session)
    try:
        assert _find_menu_action(window.window.menuBar(), "pop_eegplot:components").isEnabled()
        tools = next(action.menu() for action in window.window.menuBar().actions() if action.text() == "Tools")
        iclabel = next(action for action in tools.actions() if action.text() == "Classify components using ICLabel")
        assert iclabel.isEnabled()
    finally:
        window.window.close()


@pytest.mark.parametrize(
    "refs, expected", [([None, None], "unknown"), (["A", "B", "B"], "B"), (["B", "A"], "A"), (["", "", "A"], "unknown")]
)
def test_f2_reference_uses_most_frequent_channel_reference(refs, expected):
    eeg = _demo_eeg()
    eeg["chanlocs"] = [{"labels": str(index), "ref": ref} for index, ref in enumerate(refs)]
    assert _reference_state(eeg) == expected


@pytest.mark.parametrize("xmin,xmax,expected", [(0, 0.39, ("0.000", "0.390")), (-1, 2, ("-1", "2"))])
def test_f3_epoch_limits_share_precision(xmin, xmax, expected):
    eeg = _demo_eeg()
    eeg.update(xmin=xmin, xmax=xmax)
    rows = dict(_summary_for_session(EEGPrepSession(EEG=eeg))[2])
    assert (rows["Epoch start (sec)"].strip(), rows["Epoch end (sec)"].strip()) == expected


def test_f4_size_includes_metadata_ica_and_does_not_round_each_set():
    eeg = {"data": np.zeros(100), "icaact": np.zeros(100_000), "etc": {"notes": "x" * 200_000}}
    assert float(_size_mb(eeg)) >= 1.0
    assert float(_size_mb([{"data": np.zeros(4900)} for _ in range(10)])) == 0.4


@pytest.mark.parametrize(
    "ica,expected", [([True, True], "Yes"), ([False, False], "No"), ([False, True], "mixed, yes and no")]
)
def test_f93_summary_distinguishes_mixed_ica(ica, expected):
    session = EEGPrepSession(EEG=[_demo_eeg(ica=value) for value in ica], CURRENTSET=[1, 2])
    assert dict(_summary_for_session(session)[2])["ICA weights"] == expected

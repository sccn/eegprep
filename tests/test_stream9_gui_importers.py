"""Offscreen GUI regression checks for the Stream 9 import flows."""

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PySide6')
from PySide6 import QtCore, QtWidgets

from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.session import EEGPrepSession

pytestmark = pytest.mark.gui


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_f18_mff_picker_selects_directory(app, tmp_path):
    folder = tmp_path / 'recording.mff'
    folder.mkdir()
    dispatcher = MenuActionDispatcher(EEGPrepSession(), native_file_dialogs=False)
    observed = []

    def choose():
        dialog = app.activeModalWidget()
        if isinstance(dialog, QtWidgets.QFileDialog):
            observed.append(dialog.fileMode())
            if dialog.fileMode() == QtWidgets.QFileDialog.FileMode.Directory:
                dialog.setDirectory(str(tmp_path))
                dialog.selectFile(str(folder))
                dialog.accept()
            else:
                dialog.reject()

    timer = QtCore.QTimer()
    timer.timeout.connect(choose)
    timer.start(10)
    try:
        selected = dispatcher._open_import_filename('pop_fileio_mff', None)
    finally:
        timer.stop()
    assert observed == [QtWidgets.QFileDialog.FileMode.Directory]
    assert selected == str(folder)

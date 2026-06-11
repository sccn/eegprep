import logging
import os

import pytest

from eegprep.functions.guifunc.long_task import run_long_task


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_run_long_task_returns_result_and_forwards_progress(qapp):
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    results = []
    errors = []
    finished = []

    def task():
        logging.getLogger("eegprep.tests").info("worker progress")
        return "done"

    handle = run_long_task(
        parent=None,
        title="Running test task",
        label="Running test task.",
        task=task,
        on_success=results.append,
        on_error=errors.append,
        on_finished=lambda task_handle: (finished.append(task_handle), loop.quit()),
    )
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec()

    assert results == ["done"]
    assert errors == []
    assert finished == [handle]
    assert "worker progress" in handle.dialog.labelText()


def test_run_long_task_reports_errors(qapp):
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    results = []
    errors = []

    def task():
        raise ValueError("task failed")

    run_long_task(
        parent=None,
        title="Running test task",
        label="Running test task.",
        task=task,
        on_success=results.append,
        on_error=errors.append,
        on_finished=lambda _handle: loop.quit(),
    )
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec()

    assert results == []
    assert len(errors) == 1
    assert str(errors[0]) == "task failed"

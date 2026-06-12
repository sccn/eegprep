import logging
import os
import threading

import pytest

import eegprep.functions.guifunc.long_task as long_task_module
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
    assert "QProgressDialog" in handle.dialog.styleSheet()
    assert "QProgressBar::chunk" in handle.dialog.styleSheet()
    assert "#a8c2ff" in handle.dialog.styleSheet()
    assert "#000066" in handle.dialog.styleSheet()


def test_run_long_task_restores_eegprep_logger_level_after_forwarding_progress(qapp):
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    logger = logging.getLogger("eegprep")
    original_level = logger.level
    logger.setLevel(logging.WARNING)

    def task():
        logging.getLogger("eegprep.tests").info("worker progress")
        return "done"

    try:
        handle = run_long_task(
            parent=None,
            title="Running test task",
            label="Running test task.",
            task=task,
            on_success=lambda _result: None,
            on_error=lambda _exc: None,
            on_finished=lambda _handle: loop.quit(),
        )
        QtCore.QTimer.singleShot(3000, loop.quit)
        loop.exec()

        assert "worker progress" in handle.dialog.labelText()
        assert logger.level == logging.WARNING
    finally:
        logger.setLevel(original_level)


def test_run_long_task_callbacks_are_delivered_on_main_thread(qapp, monkeypatch):
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    main_thread_id = threading.get_ident()
    task_thread_ids = []
    message_thread_ids = []
    success_thread_ids = []
    original_update = long_task_module._update_progress_label

    def update_progress_label(progress, label, message):
        message_thread_ids.append(threading.get_ident())
        original_update(progress, label, message)

    def task():
        task_thread_ids.append(threading.get_ident())
        logging.getLogger("eegprep.tests").info("worker progress")
        return "done"

    monkeypatch.setattr(long_task_module, "_update_progress_label", update_progress_label)

    run_long_task(
        parent=None,
        title="Running test task",
        label="Running test task.",
        task=task,
        on_success=lambda _result: success_thread_ids.append(threading.get_ident()),
        on_error=lambda _exc: None,
        on_finished=lambda _handle: loop.quit(),
    )
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec()

    assert task_thread_ids
    assert task_thread_ids[0] != main_thread_id
    assert message_thread_ids
    assert all(thread_id == main_thread_id for thread_id in message_thread_ids)
    assert success_thread_ids == [main_thread_id]


def test_run_long_task_reports_errors(qapp):
    from PySide6 import QtCore

    loop = QtCore.QEventLoop()
    results = []
    errors = []
    error_thread_ids = []
    main_thread_id = threading.get_ident()

    def task():
        raise ValueError("task failed")

    def on_error(exc):
        error_thread_ids.append(threading.get_ident())
        errors.append(exc)

    run_long_task(
        parent=None,
        title="Running test task",
        label="Running test task.",
        task=task,
        on_success=results.append,
        on_error=on_error,
        on_finished=lambda _handle: loop.quit(),
    )
    QtCore.QTimer.singleShot(3000, loop.quit)
    loop.exec()

    assert results == []
    assert len(errors) == 1
    assert str(errors[0]) == "task failed"
    assert error_thread_ids == [main_thread_id]

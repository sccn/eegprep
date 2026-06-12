"""Qt helper for running long GUI actions off the UI thread."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import threading
from typing import Any

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtWidgets = None


@dataclass
class LongTaskHandle:
    """Keep Qt task objects alive until their worker thread finishes."""

    thread: Any
    worker: Any
    dialog: Any
    receiver: Any | None = None


_LOGGER_LOCK = threading.Lock()
_LOGGER_DEPTH = 0
_LOGGER_OLD_LEVEL: int | None = None


def run_long_task(
    *,
    parent: Any | None,
    title: str,
    label: str,
    task: Callable[[], Any],
    on_success: Callable[[Any], None],
    on_error: Callable[[Exception], None],
    on_finished: Callable[[LongTaskHandle], None] | None = None,
) -> LongTaskHandle:
    """Run ``task`` in a Qt worker thread with an indeterminate progress dialog."""
    qt_core, qt_widgets = _require_qt()

    progress = qt_widgets.QProgressDialog(label, None, 0, 0, parent)
    progress.setWindowTitle(title)
    progress.setCancelButton(None)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setMinimumDuration(0)
    progress.setWindowModality(qt_core.Qt.WindowModal)

    class Worker(qt_core.QObject):
        succeeded = qt_core.Signal(object)
        failed = qt_core.Signal(object)
        message = qt_core.Signal(str)
        finished = qt_core.Signal()

        def run(self) -> None:
            handler = _SignalLogHandler(self.message)
            handler.setFormatter(logging.Formatter("%(message)s"))
            with _ForwardEegprepLogs(handler):
                self.message.emit(label)
                try:
                    result = task()
                    self.succeeded.emit(result)
                except Exception as exc:  # noqa: BLE001 - forwarded to GUI error handler.
                    self.failed.emit(exc)
                finally:
                    self.finished.emit()

    thread = qt_core.QThread()
    worker = Worker()

    class Receiver(qt_core.QObject):
        @qt_core.Slot(str)
        def handle_message(self, message: str) -> None:
            _update_progress_label(progress, label, message)

        @qt_core.Slot(object)
        def handle_success(self, result: Any) -> None:
            on_success(result)

        @qt_core.Slot(object)
        def handle_error(self, exc: Exception) -> None:
            on_error(exc)

        @qt_core.Slot()
        def handle_finished(self) -> None:
            progress.close()
            if on_finished is not None:
                on_finished(handle)

    receiver = Receiver()
    handle = LongTaskHandle(thread=thread, worker=worker, dialog=progress, receiver=receiver)

    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.message.connect(receiver.handle_message)
    worker.succeeded.connect(receiver.handle_success)
    worker.failed.connect(receiver.handle_error)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.finished.connect(receiver.handle_finished)
    thread.finished.connect(receiver.deleteLater)

    progress._eegprep_long_task = handle
    progress.show()
    qt_core.QTimer.singleShot(0, thread.start)
    return handle


class _SignalLogHandler(logging.Handler):
    def __init__(self, signal: Any):
        super().__init__(level=logging.INFO)
        self.signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.signal.emit(self.format(record))
        except Exception:
            self.handleError(record)


class _ForwardEegprepLogs:
    def __init__(self, handler: logging.Handler) -> None:
        self.handler = handler
        self.logger = logging.getLogger("eegprep")

    def __enter__(self) -> None:
        global _LOGGER_DEPTH, _LOGGER_OLD_LEVEL
        with _LOGGER_LOCK:
            if _LOGGER_DEPTH == 0:
                _LOGGER_OLD_LEVEL = self.logger.level
                if self.logger.level == logging.NOTSET or self.logger.level > logging.INFO:
                    self.logger.setLevel(logging.INFO)
            _LOGGER_DEPTH += 1
            self.logger.addHandler(self.handler)

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        global _LOGGER_DEPTH, _LOGGER_OLD_LEVEL
        with _LOGGER_LOCK:
            self.logger.removeHandler(self.handler)
            _LOGGER_DEPTH -= 1
            if _LOGGER_DEPTH == 0:
                self.logger.setLevel(logging.NOTSET if _LOGGER_OLD_LEVEL is None else _LOGGER_OLD_LEVEL)
                _LOGGER_OLD_LEVEL = None


def _update_progress_label(progress: Any, label: str, message: str) -> None:
    message = str(message).strip()
    progress.setLabelText(label if not message or message == label else f"{label}\n{message}")


def _require_qt() -> tuple[Any, Any]:
    if QtCore is None or QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI progress dialogs. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtCore, QtWidgets

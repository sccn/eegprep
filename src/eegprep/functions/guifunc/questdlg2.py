"""EEGLAB ``questdlg2``-style question dialog."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtWidgets = None


_MAIN_WINDOW_OBJECT_NAME = "EEGPrep"


def questdlg2(prompt: str, title: str, *buttons: str, parent: Any | None = None) -> str:
    """Ask a question and return the clicked button label.

    The last value in ``buttons`` is the default, matching ``questdlg2.m``.
    Closing the dialog returns an empty string.
    """
    if QtCore is None or QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep question dialogs. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    if len(buttons) < 2:
        raise ValueError("questdlg2 requires button labels and a default")
    labels = buttons[:-1]
    default = buttons[-1]
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QDialog(_dialog_parent(parent))
    dialog.setObjectName("questdlg2")
    dialog.setWindowTitle(title)
    dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
    layout = QtWidgets.QVBoxLayout(dialog)
    message = QtWidgets.QLabel(prompt, dialog)
    message.setObjectName("questdlg2_prompt")
    message.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    layout.addWidget(message)
    clicked = {"": ""}
    button_row = QtWidgets.QHBoxLayout()
    button_row.addStretch(1)
    for label in labels:
        button = QtWidgets.QPushButton(label, dialog)
        if label == default:
            button.setDefault(True)
        button_row.addWidget(button)

        def choose(_checked: bool = False, label: str = label) -> None:
            clicked[""] = label
            dialog.accept()

        button.clicked.connect(choose)
    layout.addLayout(button_row)
    dialog.exec()
    return clicked[""]


def _dialog_parent(parent: Any | None) -> Any | None:
    if parent is not None or QtWidgets is None:
        return parent
    app = QtWidgets.QApplication.instance()
    if not isinstance(app, QtWidgets.QApplication):
        return None
    for widget in app.topLevelWidgets():
        if widget.objectName() == _MAIN_WINDOW_OBJECT_NAME and widget.isVisible():
            return widget
    return None

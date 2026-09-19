"""EEGLAB ``errordlg2``-style error dialog."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtWidgets
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtWidgets = None


def errordlg2(prompt: str, title: str = "Error", parent: Any | None = None) -> int:
    """Display a modal error message and return the Qt dialog result."""
    _app, dialog = build_errordlg2(prompt, title, parent)
    return int(dialog.exec())


def build_errordlg2(prompt: str, title: str = "Error", parent: Any | None = None) -> tuple[Any, Any]:
    """Build an error dialog without executing it, for GUI tests and capture."""
    if QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI error dialogs. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dialog = QtWidgets.QMessageBox(parent)
    dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    dialog.setWindowTitle(title)
    dialog.setText(prompt)
    dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
    dialog.setObjectName("errordlg2")
    return app, dialog

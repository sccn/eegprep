"""Shared EEGPrep Qt style fragments."""

from __future__ import annotations

from typing import Any


EEGLAB_BACKGROUND = "#a8c2ff"
EEGLAB_TEXT = "#000066"
EEGLAB_DISABLED_TEXT = "#7c86a8"
EEGLAB_CONTROL_BACKGROUND = "#ffffff"
EEGLAB_CONTROL_DISABLED_BACKGROUND = "#dce6ff"
EEGLAB_CONTROL_BORDER = "#7f7f7f"
EEGLAB_BUTTON_BACKGROUND = "#eeeeee"
EEGLAB_SELECTION_BACKGROUND = "#c6d9ff"
EEGLAB_PROGRESS_CHUNK = "#0078d7"


def eeglab_floating_dialog_stylesheet() -> str:
    """Return styles for child dialogs that must stay legible in dark mode."""
    return f"""
        QFileDialog, QProgressDialog {{
            background: {EEGLAB_BACKGROUND};
            color: {EEGLAB_TEXT};
        }}
        QFileDialog QLabel,
        QFileDialog QCheckBox,
        QFileDialog QRadioButton,
        QProgressDialog QLabel {{
            background: transparent;
            color: {EEGLAB_TEXT};
        }}
        QFileDialog QLabel:disabled,
        QFileDialog QCheckBox:disabled,
        QFileDialog QRadioButton:disabled,
        QProgressDialog QLabel:disabled {{
            color: {EEGLAB_DISABLED_TEXT};
        }}
        QFileDialog QLineEdit,
        QFileDialog QTextEdit,
        QFileDialog QPlainTextEdit,
        QFileDialog QComboBox,
        QFileDialog QListView,
        QFileDialog QTreeView,
        QFileDialog QTableView,
        QFileDialog QHeaderView::section {{
            background: {EEGLAB_CONTROL_BACKGROUND};
            border: 1px solid {EEGLAB_CONTROL_BORDER};
            color: {EEGLAB_TEXT};
            selection-background-color: {EEGLAB_SELECTION_BACKGROUND};
            selection-color: {EEGLAB_TEXT};
        }}
        QFileDialog QLineEdit:disabled,
        QFileDialog QTextEdit:disabled,
        QFileDialog QPlainTextEdit:disabled,
        QFileDialog QComboBox:disabled,
        QFileDialog QListView:disabled,
        QFileDialog QTreeView:disabled,
        QFileDialog QTableView:disabled {{
            background: {EEGLAB_CONTROL_DISABLED_BACKGROUND};
            color: {EEGLAB_DISABLED_TEXT};
        }}
        QFileDialog QPushButton,
        QProgressDialog QPushButton {{
            background: {EEGLAB_BUTTON_BACKGROUND};
            border: 1px solid {EEGLAB_CONTROL_BORDER};
            color: {EEGLAB_TEXT};
            padding: 2px 10px;
        }}
        QFileDialog QPushButton:disabled,
        QProgressDialog QPushButton:disabled {{
            color: {EEGLAB_DISABLED_TEXT};
        }}
        QProgressBar {{
            background: {EEGLAB_CONTROL_DISABLED_BACKGROUND};
            border: 1px solid {EEGLAB_CONTROL_BORDER};
            color: {EEGLAB_TEXT};
            min-height: 12px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background: {EEGLAB_PROGRESS_CHUNK};
        }}
        """


def append_eeglab_floating_dialog_style(widget: Any) -> None:
    """Append floating-dialog styles to a Qt widget without replacing local rules."""
    stylesheet = widget.styleSheet()
    widget.setStyleSheet(stylesheet + "\n" + eeglab_floating_dialog_stylesheet())

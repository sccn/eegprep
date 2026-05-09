"""EEGLAB ``listdlg2``-style Qt list dialog."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:  # pragma: no cover - depends on optional GUI dependency
    from PySide6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - depends on optional GUI dependency
    QtCore = None
    QtWidgets = None


def _require_qt() -> tuple[Any, Any]:
    if QtCore is None or QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI list dialogs. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtCore, QtWidgets


def listdlg2(
    *,
    promptstring: str = "",
    liststring: Sequence[str],
    selectionmode: str = "multiple",
    initialvalue: Sequence[int] | None = None,
    listsize: tuple[int, int] | None = None,
    name: str = "",
    parent: Any | None = None,
) -> tuple[list[int], int, str]:
    """Open an EEGLAB-like list selector.

    Returns 1-based selected list positions, an OK flag, and the selected
    display strings joined by spaces, matching EEGLAB ``listdlg2``.
    """
    qt_core, qt_widgets = _require_qt()

    dialog, list_widget, list_items = _create_dialog(
        qt_core,
        qt_widgets,
        promptstring=promptstring,
        liststring=liststring,
        selectionmode=selectionmode,
        initialvalue=initialvalue,
        listsize=listsize,
        name=name,
        parent=parent,
    )

    if dialog.exec() != qt_widgets.QDialog.Accepted:
        return [], 0, ""
    selected = sorted(item.data(qt_core.Qt.UserRole) for item in list_widget.selectedItems())
    selected_strings = [list_items[index - 1] for index in selected]
    return selected, 1, " ".join(selected_strings)


def build_listdlg2_dialog(
    *,
    promptstring: str = "",
    liststring: Sequence[str],
    selectionmode: str = "multiple",
    initialvalue: Sequence[int] | None = None,
    listsize: tuple[int, int] | None = None,
    name: str = "",
    parent: Any | None = None,
) -> tuple[Any, Any]:
    """Build a listdlg2 dialog without executing it, for visual capture tests."""
    qt_core, qt_widgets = _require_qt()

    app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    dialog, _list_widget, _list_items = _create_dialog(
        qt_core,
        qt_widgets,
        promptstring=promptstring,
        liststring=liststring,
        selectionmode=selectionmode,
        initialvalue=initialvalue,
        listsize=listsize,
        name=name,
        parent=parent,
    )
    return app, dialog


def _create_dialog(
    QtCore: Any,
    QtWidgets: Any,
    *,
    promptstring: str,
    liststring: Sequence[str],
    selectionmode: str,
    initialvalue: Sequence[int] | None,
    listsize: tuple[int, int] | None,
    name: str,
    parent: Any | None,
) -> tuple[Any, Any, list[str]]:
    list_items = [str(item) for item in liststring]
    initial = _normalise_initial(initialvalue, len(list_items), selectionmode)
    dialog = QtWidgets.QDialog(parent)
    dialog.setObjectName("listdlg2")
    dialog.setWindowTitle(name)
    _apply_listdlg_style(dialog)
    visible_rows = min(max(len(list_items), 1), 10)
    if listsize is not None:
        width, height = listsize
    else:
        width, height = 176, 115 + visible_rows * 19
    dialog.resize(width, height)

    if promptstring:
        label = QtWidgets.QLabel(promptstring, dialog)
        label.setObjectName("prompt")
        label.setAlignment(QtCore.Qt.AlignLeft)
        label.setGeometry(18, 15, width - 36, 40)

    list_widget = QtWidgets.QListWidget(dialog)
    list_widget.setObjectName("listboxvals")
    if selectionmode.lower() == "single" or len(list_items) == 1:
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    else:
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    for index, item_text in enumerate(list_items, start=1):
        item = QtWidgets.QListWidgetItem(item_text)
        item.setData(QtCore.Qt.UserRole, index)
        if index in initial:
            item.setSelected(True)
        list_widget.addItem(item)
    list_widget.setGeometry(18, 64 if promptstring else 15, width - 36, visible_rows * 20 + 8)

    cancel = QtWidgets.QPushButton("Cancel", dialog)
    ok = QtWidgets.QPushButton("Ok", dialog)
    cancel.setObjectName("cancel")
    ok.setObjectName("ok")
    cancel.clicked.connect(dialog.reject)
    ok.clicked.connect(dialog.accept)
    button_y = height - 33
    cancel.setGeometry(18, button_y, 62, 18)
    ok.setGeometry(width - 80, button_y, 62, 18)
    return dialog, list_widget, list_items


def _normalise_initial(
    initialvalue: Sequence[int] | None,
    list_length: int,
    selectionmode: str,
) -> set[int]:
    if initialvalue is None:
        initial = [] if selectionmode.lower() == "multiple" else [1]
    elif isinstance(initialvalue, (int, float)):
        initial = [int(initialvalue)]
    else:
        initial = [int(value) for value in initialvalue]
    return {value for value in initial if 1 <= value <= list_length}


def _apply_listdlg_style(dialog: Any) -> None:
    dialog.setStyleSheet(
        """
        QDialog {
            background: #a8c2ff;
            color: #000066;
            font-size: 16px;
        }
        QLabel {
            color: #000066;
            background: transparent;
            font-size: 16px;
        }
        QListWidget {
            background: white;
            color: black;
            border: 1px solid #7f7f7f;
            font-size: 16px;
        }
        QPushButton {
            background: #eeeeee;
            border: 1px solid #7f7f7f;
            min-width: 62px;
            max-width: 62px;
            min-height: 18px;
            max-height: 18px;
            padding: 0;
            color: #000066;
            font-size: 16px;
        }
        """
    )

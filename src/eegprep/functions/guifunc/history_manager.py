"""Interactive History Manager sidebar."""

from __future__ import annotations

import re

from eegprep.functions.guifunc.session import EEGPrepSession

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    QtCore = None
    QtGui = None
    QtWidgets = None


def _extract_function_name(command: str) -> str:
    match = re.search(r'=\s*([a-zA-Z_]\w*)\s*\(', command)
    if match:
        return match.group(1)
    match = re.search(r'^([a-zA-Z_]\w*)\s*\(', command)
    if match:
        return match.group(1)
    return "Command"


class HistoryManagerWidget(QtWidgets.QDockWidget):
    """Sidebar for viewing and grouping command history."""

    def __init__(self, session: EEGPrepSession, parent: QtWidgets.QWidget | None = None):
        super().__init__("Interactive History", parent)
        self.session = session
        self.setObjectName("HistoryManagerWidget")

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        # Reordering and deleting are removed to preserve scientific audit history (append-only)
        layout.addWidget(self.tree)

        self.setWidget(widget)

        self.session.add_change_listener(self._on_session_changed)
        self._on_session_changed(self.session)

    def _on_session_changed(self, session: EEGPrepSession) -> None:
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self.tree.clear()

        current_func = None
        current_group: QtWidgets.QTreeWidgetItem | None = None

        for cmd in self.session.ALLCOM:
            func_name = _extract_function_name(cmd)
            if func_name == current_func and current_group is not None:
                item = QtWidgets.QTreeWidgetItem(current_group, [cmd])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, cmd)
            else:
                current_func = func_name
                current_group = QtWidgets.QTreeWidgetItem(self.tree, [func_name])
                current_group.setExpanded(True)
                item = QtWidgets.QTreeWidgetItem(current_group, [cmd])
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, cmd)

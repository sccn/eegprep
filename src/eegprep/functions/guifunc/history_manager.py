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
    """Sidebar for staging and committing command history."""

    def __init__(self, session: EEGPrepSession, parent: QtWidgets.QWidget | None = None):
        super().__init__("Interactive History", parent)
        self.session = session
        self.session.HISTORY_STAGING_ENABLED = True
        self.setObjectName("HistoryManagerWidget")

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        layout.addWidget(self.tree)

        btn_layout = QtWidgets.QHBoxLayout()
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self._delete_selected)
        btn_layout.addWidget(self.delete_btn)

        self.commit_btn = QtWidgets.QPushButton("Commit")
        self.commit_btn.clicked.connect(self._commit)
        btn_layout.addWidget(self.commit_btn)

        layout.addLayout(btn_layout)
        self.setWidget(widget)

        self.session.add_change_listener(self._on_session_changed)
        self._on_session_changed(self.session)

    def _on_session_changed(self, session: EEGPrepSession) -> None:
        # Avoid rebuilding if user is interacting
        if self.tree.hasFocus():
            return
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        self.tree.clear()
        if not hasattr(self.session, "STAGEDCOM"):
            return

        current_func = None
        current_group: QtWidgets.QTreeWidgetItem | None = None

        for cmd in getattr(self.session, "STAGEDCOM", []):
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

    def _delete_selected(self) -> None:
        if not hasattr(self.session, "STAGEDCOM"):
            return
        selected = self.tree.selectedItems()
        to_delete = []
        for item in selected:
            cmd = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if cmd:
                to_delete.append(cmd)
            else:
                # Group selected, delete all children
                for i in range(item.childCount()):
                    child = item.child(i)
                    cmd = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if cmd:
                        to_delete.append(cmd)

        staged = getattr(self.session, "STAGEDCOM", [])
        for cmd in to_delete:
            if cmd in staged:
                staged.remove(cmd)
        self.session.notify_changed()

    def _commit(self) -> None:
        if not hasattr(self.session, "STAGEDCOM"):
            return
        selected = self.tree.selectedItems()
        to_commit = []
        for item in selected:
            cmd = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if cmd:
                to_commit.append(cmd)
            else:
                for i in range(item.childCount()):
                    child = item.child(i)
                    cmd = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if cmd:
                        to_commit.append(cmd)

        # Deduplicate while preserving order
        unique_to_commit = list(dict.fromkeys(to_commit))

        if not unique_to_commit:
            return

        self.session.commit_staged_commands(unique_to_commit)

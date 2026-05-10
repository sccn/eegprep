"""Action dispatch for the EEGPrep EEGLAB-style main window."""

from __future__ import annotations

import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eegprep.functions.guifunc.menu_placeholders import is_placeholder_action, placeholder_message
from eegprep.functions.guifunc.pophelp import pophelp
from eegprep.functions.guifunc.session import EEGPrepSession, has_eeg_data
from eegprep.functions.popfunc._coming_soon import coming_soon
from eegprep.functions.popfunc.pop_adjustevents import pop_adjustevents
from eegprep.functions.popfunc.pop_interp import pop_interp
from eegprep.functions.popfunc.pop_loadset import pop_loadset
from eegprep.functions.popfunc.pop_reref import pop_reref
from eegprep.functions.popfunc.pop_saveset import pop_saveset
from eegprep.plugins.ICLabel.iclabel import iclabel


IMPLEMENTED_ACTIONS = {
    "clear_study",
    "help",
    "mailto",
    "pop_adjustevents",
    "pop_delset",
    "pop_iclabel",
    "pop_interp",
    "pop_loadset",
    "pop_reref",
    "pop_saveset",
    "quit",
    "retrieve_dataset",
    "tutorial",
}


class MenuActionDispatcher:
    """Dispatch menu action identifiers to real functions or placeholders."""

    def __init__(self, session: EEGPrepSession, refresh: Callable[[], None] | None = None):
        self.session = session
        self.refresh = refresh

    def dispatch_gui(self, action: str, parent: Any | None = None) -> None:
        """Run a menu action from Qt and show user-facing errors."""
        try:
            self.dispatch(action, parent)
        except Exception as exc:
            self._warn(parent, str(exc))

    def dispatch(self, action: str, parent: Any | None = None) -> None:
        """Run a menu action."""
        base, _sep, variant = action.partition(":")
        if base == "quit":
            if parent is not None:
                parent.close()
            return
        if base == "clear_study":
            self.session.clear_all()
            self._refresh()
            return
        if base == "help":
            self._show_help(variant or "eeglab", parent)
            return
        if base == "mailto":
            webbrowser.open(f"mailto:{variant}")
            return
        if base == "tutorial":
            webbrowser.open("https://eeglab.org/tutorials/")
            return
        if base == "pop_loadset":
            self._loadset(parent)
            return
        if base == "pop_saveset":
            self._saveset(parent, resave=variant == "resave")
            return
        if base == "pop_delset":
            self.session.delete_current()
            self._refresh()
            return
        if base == "retrieve_dataset":
            self._retrieve_dataset(int(variant))
            return
        if base == "pop_adjustevents":
            self._run_pop_function("pop_adjustevents", parent)
            return
        if base == "pop_reref":
            self._run_pop_function("pop_reref", parent)
            return
        if base == "pop_interp":
            self._run_pop_function("pop_interp", parent)
            return
        if base == "pop_iclabel":
            self._run_iclabel(parent)
            return
        self.show_coming_soon(action, parent)

    def show_coming_soon(self, action: str, parent: Any | None = None) -> None:
        """Show the shared placeholder dialog."""
        message = placeholder_message(action)
        qt_widgets = _qt_widgets()
        if qt_widgets is None:
            coming_soon(action)
            return
        qt_widgets.QMessageBox.information(parent, "EEGPrep", message)

    def _loadset(self, parent: Any | None) -> None:
        qt_widgets = _require_qt_widgets()
        filename, _filter = qt_widgets.QFileDialog.getOpenFileName(
            parent,
            "Load existing dataset",
            "",
            "EEGLAB datasets (*.set *.mat);;All files (*)",
        )
        if not filename:
            return
        eeg = pop_loadset(filename)
        self.session.store_current(eeg, new=True, command=f"EEG = pop_loadset({filename!r});")
        self._refresh()

    def _saveset(self, parent: Any | None, *, resave: bool = False) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=resave)
        if selection is None:
            return
        datasets = selection if isinstance(selection, list) else [selection]
        filenames = [_existing_dataset_filename(eeg) if resave else "" for eeg in datasets]
        if resave and len(datasets) > 1 and not all(filenames):
            self._warn(parent, "Cannot resave multiple datasets until every selected dataset has a filename.")
            return
        filename = filenames[0] if len(datasets) == 1 else ""
        if len(datasets) == 1 and not filename:
            qt_widgets = _require_qt_widgets()
            filename, _filter = qt_widgets.QFileDialog.getSaveFileName(
                parent,
                "Save current dataset as",
                str(datasets[0].get("filename") or ""),
                "EEGLAB datasets (*.set);;All files (*)",
            )
            filenames = [filename]
        if len(datasets) == 1 and not filename:
            return
        for eeg, filename in zip(datasets, filenames):
            pop_saveset(eeg, filename)
            _apply_save_metadata(eeg, filename)
        stored = datasets if isinstance(selection, list) else datasets[0]
        command = (
            "EEG = pop_saveset(EEG, 'savemode', 'resave');"
            if resave
            else f"EEG = pop_saveset(EEG, {filenames[0]!r});"
        )
        self.session.store_current(stored, command=command, mark_saved=True)
        self._refresh()

    def _run_pop_function(self, name: str, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=name == "pop_reref")
        if selection is None:
            return
        if name == "pop_adjustevents":
            out = pop_adjustevents(selection, return_com=True)
        elif name == "pop_reref":
            out = pop_reref(selection, return_com=True)
        elif name == "pop_interp":
            out = pop_interp(selection, alleeg=self.session.ALLEEG, return_com=True)
        else:
            self.show_coming_soon(name, parent)
            return
        if isinstance(out, tuple):
            eeg_out, command = out[0], out[1] if len(out) > 1 else ""
        else:
            eeg_out, command = out, ""
        if command:
            self.session.store_current(eeg_out, command=command)
            self._refresh()

    def _run_iclabel(self, parent: Any | None) -> None:
        selection = self._current_selection_or_warn(parent, allow_multiple=True)
        if selection is None:
            return
        if isinstance(selection, list):
            self.session.store_current(
                [iclabel(eeg) for eeg in selection],
                command="EEG = pop_iclabel(EEG, 'default');",
            )
        else:
            self.session.store_current(iclabel(selection), command="EEG = pop_iclabel(EEG, 'default');")
        self._refresh()

    def _retrieve_dataset(self, index: int) -> None:
        was_study = self.session.CURRENTSTUDY == 1
        self.session.retrieve(index)
        command = f"[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, 'retrieve', {index});"
        if was_study:
            self.session.CURRENTSTUDY = 0
            command = f"CURRENTSTUDY = 0;{command}"
        self.session.add_history(command)
        self._refresh()

    def _show_help(self, function_name: str, parent: Any | None) -> None:
        try:
            pophelp(function_name, parent=parent)
        except Exception:
            self.show_coming_soon(f"pophelp:{function_name}", parent)

    def _current_selection_or_warn(
        self,
        parent: Any | None,
        *,
        allow_multiple: bool = False,
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        selection = self.session.current_eeg()
        if isinstance(selection, list):
            if not any(has_eeg_data(eeg) for eeg in selection):
                self._warn(parent, "No current dataset")
                return None
            if len(selection) > 1:
                if allow_multiple:
                    return selection
                self._warn(parent, "This action is not available for multiple selected datasets")
                return None
            return selection[0]
        if has_eeg_data(selection):
            return selection
        self._warn(parent, "No current dataset")
        return None

    def _warn(self, parent: Any | None, message: str) -> None:
        qt_widgets = _qt_widgets()
        if qt_widgets is not None:
            qt_widgets.QMessageBox.warning(parent, "EEGPrep", message)

    def _refresh(self) -> None:
        if self.refresh is not None:
            self.refresh()


def _existing_dataset_filename(eeg: dict[str, Any]) -> str:
    filepath = str(eeg.get("filepath") or "")
    filename = str(eeg.get("filename") or "")
    if filepath and filename:
        return str(Path(filepath) / filename)
    return filename


def _apply_save_metadata(eeg: dict[str, Any], filename: str) -> None:
    path = Path(filename)
    eeg["filename"] = path.name
    eeg["filepath"] = str(path.parent)
    eeg["saved"] = "yes"


def action_kind(action: str) -> str:
    """Return ``implemented``, ``placeholder``, or ``unknown`` for an action id."""
    base = action.partition(":")[0]
    if base in IMPLEMENTED_ACTIONS:
        return "implemented"
    if is_placeholder_action(action):
        return "placeholder"
    return "unknown"


def _qt_widgets() -> Any | None:
    try:
        from PySide6 import QtWidgets
    except ImportError:
        return None
    return QtWidgets


def _require_qt_widgets() -> Any:
    qt_widgets = _qt_widgets()
    if qt_widgets is None:
        raise RuntimeError(
            "PySide6 is required for EEGPrep GUI actions. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return qt_widgets

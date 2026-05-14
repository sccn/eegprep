"""PySide6 EEGLAB-style main window for EEGPrep."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.eeglab_menu import eeglab_menus
from eegprep.functions.guifunc.menu_actions import MenuActionDispatcher
from eegprep.functions.guifunc.menu_spec import MenuItemSpec, menu_enabled
from eegprep.functions.guifunc.session import EEGPrepSession, has_eeg_data

try:  # pragma: no cover - optional GUI dependency
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:  # pragma: no cover - optional GUI dependency
    QtCore = None
    QtGui = None
    QtWidgets = None


BACKEEGLABCOLOR = "#a8c2ff"
GUITEXTCOLOR = "#000066"
PLUGINMENUCOLOR = "#800080"


def _require_qt() -> tuple[Any, Any, Any]:
    if QtCore is None or QtGui is None or QtWidgets is None:
        raise RuntimeError(
            "PySide6 is required for the EEGPrep main window. Install it with "
            "`pip install -e .[gui]` or `pip install eegprep[gui]`."
        )
    return QtCore, QtGui, QtWidgets


class EEGPrepMainWindow:
    """EEGLAB-like main window backed by an :class:`EEGPrepSession`."""

    def __init__(
        self,
        session: EEGPrepSession | None = None,
        *,
        all_menus: bool | None = None,
        include_plugins: bool = True,
        native_menu_bar: bool | None = None,
    ) -> None:
        qt_core, qt_gui, qt_widgets = _require_qt()
        self._qt_core = qt_core
        self._qt_gui = qt_gui
        self._qt_widgets = qt_widgets
        self.app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
        self.session = session or EEGPrepSession()
        self.all_menus = bool(EEG_OPTIONS.get("option_allmenus", 0)) if all_menus is None else bool(all_menus)
        self.include_plugins = include_plugins
        self.window = qt_widgets.QMainWindow()
        self.window.setObjectName("EEGPrep")
        self.window.setWindowTitle("EEGPrep")
        self.window.resize(520, 380)
        self.window.setMinimumSize(460, 340)
        self.window.setStyleSheet(_main_window_stylesheet())
        use_native_menu_bar = sys.platform == "darwin" if native_menu_bar is None else bool(native_menu_bar)
        self.window.menuBar().setNativeMenuBar(use_native_menu_bar)
        self.dispatcher = MenuActionDispatcher(self.session, refresh=self.refresh)
        self._build_central_widget()
        self.refresh()

    def show(self) -> "EEGPrepMainWindow":
        """Show the main window and return ``self``."""
        self.window.show()
        self.window.raise_()
        return self

    def exec(self) -> int:
        """Show the window and run the Qt event loop."""
        self.show()
        return self.app.exec()

    def refresh(self) -> None:
        """Refresh menus and summary text from session state."""
        self._build_menus()
        self._update_summary()

    def menu_inventory(self) -> list[dict[str, Any]]:
        """Return an inventory of the rendered Qt menu tree."""
        return [_action_inventory(action) for action in self.window.menuBar().actions()]

    def open_menu(self, label: str) -> Any:
        """Open a top-level menu for deterministic screenshot capture."""
        for action in self.window.menuBar().actions():
            if action.text() == label and action.menu() is not None:
                menu = action.menu()
                pos = self.window.menuBar().actionGeometry(action).bottomLeft()
                menu.popup(self.window.menuBar().mapToGlobal(pos))
                self.app.processEvents()
                return menu
        raise ValueError(f"No top-level menu labeled {label!r}")

    def _build_central_widget(self) -> None:
        qt_core, _qt_gui, qt_widgets = self._qt_core, self._qt_gui, self._qt_widgets
        central = qt_widgets.QWidget()
        central.setObjectName("main_panel")
        outer = qt_widgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        frame = qt_widgets.QFrame()
        frame.setObjectName("eegprep_frame")
        frame.setFrameShape(qt_widgets.QFrame.Box)
        frame.setFrameShadow(qt_widgets.QFrame.Plain)
        frame_layout = qt_widgets.QVBoxLayout(frame)
        frame_layout.setContentsMargins(30, 4, 14, 6)
        frame_layout.setSpacing(1)

        self.heading = qt_widgets.QLabel("No current dataset")
        self.heading.setObjectName("win0")
        self.heading.setAlignment(qt_core.Qt.AlignLeft)
        _configure_eeglab_label(self.heading, qt_widgets)
        frame_layout.addWidget(self.heading)

        self.stack = qt_widgets.QStackedWidget()
        frame_layout.addWidget(self.stack, 1)

        startup = qt_widgets.QWidget()
        startup_layout = qt_widgets.QVBoxLayout(startup)
        startup_layout.setContentsMargins(0, 3, 0, 0)
        startup_layout.setSpacing(0)
        self.startup_labels = []
        for index, line in enumerate(_startup_lines()):
            label = qt_widgets.QLabel(line)
            label.setObjectName("startup_title" if index == 0 else "startup_line")
            _configure_eeglab_label(label, qt_widgets)
            self.startup_labels.append(label)
            startup_layout.addWidget(label)
        startup_layout.addStretch(1)
        self.stack.addWidget(startup)

        data_panel = qt_widgets.QWidget()
        data_layout = qt_widgets.QGridLayout(data_panel)
        data_layout.setContentsMargins(0, 3, 0, 0)
        data_layout.setHorizontalSpacing(10)
        data_layout.setVerticalSpacing(0)
        self.file_label = qt_widgets.QLabel("")
        self.file_label.setObjectName("win1")
        _configure_eeglab_label(self.file_label, qt_widgets)
        data_layout.addWidget(self.file_label, 0, 0, 1, 2)
        self.name_labels = []
        self.value_labels = []
        for row in range(12):
            name_label = qt_widgets.QLabel("")
            value_label = qt_widgets.QLabel("")
            name_label.setObjectName(f"win{row + 2}")
            value_label.setObjectName(f"val{row + 2}")
            _configure_eeglab_label(name_label, qt_widgets)
            _configure_eeglab_label(value_label, qt_widgets)
            self.name_labels.append(name_label)
            self.value_labels.append(value_label)
            data_layout.addWidget(name_label, row + 1, 0)
            data_layout.addWidget(value_label, row + 1, 1)
        data_layout.setColumnStretch(0, 1)
        data_layout.setColumnStretch(1, 1)
        self.stack.addWidget(data_panel)

        outer.addWidget(frame)
        self.window.setCentralWidget(central)

    def _build_menus(self) -> None:
        menubar = self.window.menuBar()
        menubar.clear()
        statuses = self.session.menu_statuses()
        for spec in self._current_menu_specs():
            action = self._add_top_menu(menubar, spec, statuses)
            if spec.label == "Study" and "multiple_datasets" in statuses:
                action.setEnabled(False)

    def _current_menu_specs(self) -> tuple[MenuItemSpec, ...]:
        specs = []
        for spec in eeglab_menus(all_menus=self.all_menus, include_plugins=self.include_plugins):
            if spec.label == "Datasets":
                spec = spec.with_children(self._dataset_menu_items())
            specs.append(spec)
        return tuple(specs)

    def _dataset_menu_items(self) -> tuple[MenuItemSpec, ...]:
        summaries = self.session.dataset_summaries()
        if not summaries:
            return (MenuItemSpec("Select multiple datasets", action="select_multiple_datasets", separator=True),)
        items = [
            MenuItemSpec(
                label,
                action=f"retrieve_dataset:{index}",
                userdata="study:on",
                enabled=True,
                checked=_selected,
            )
            for index, label, _selected in summaries
        ]
        if len(summaries) > 1:
            items.append(MenuItemSpec("Select multiple datasets", action="select_multiple_datasets", userdata="study:on", separator=True))
        if self.session.CURRENTSTUDY == 1 and self.session.STUDY:
            items.append(MenuItemSpec("Select the study set", action="select_study_set", userdata="study:on", separator=True))
        return tuple(items)

    def _add_top_menu(self, menubar: Any, spec: MenuItemSpec, statuses: set[str]) -> Any:
        menu = menubar.addMenu(spec.label)
        action = menu.menuAction()
        _apply_action_metadata(action, spec, self._qt_gui)
        action.setEnabled(menu_enabled(spec, statuses))
        if spec.tag:
            menu.setObjectName(spec.tag)
        for child in spec.children:
            self._add_menu_item(menu, child, statuses)
        return action

    def _add_menu_item(self, menu: Any, spec: MenuItemSpec, statuses: set[str]) -> Any:
        if spec.separator:
            menu.addSeparator()
        if spec.children:
            submenu = menu.addMenu(spec.label)
            if spec.tag:
                submenu.setObjectName(spec.tag)
            action = submenu.menuAction()
            _apply_action_metadata(action, spec, self._qt_gui)
            action.setEnabled(menu_enabled(spec, statuses))
            if spec.origin != "core":
                action.setIconText(spec.label)
            for child in spec.children:
                self._add_menu_item(submenu, child, statuses)
            return action
        action = menu.addAction(spec.label)
        _apply_action_metadata(action, spec, self._qt_gui)
        action.setEnabled(menu_enabled(spec, statuses))
        if spec.checked:
            action.setCheckable(True)
            action.setChecked(True)
        if spec.action:
            action.triggered.connect(lambda _checked=False, action_id=spec.action: self.dispatcher.dispatch_gui(action_id, self.window))
        if spec.origin != "core":
            action.setProperty("eegprep_plugin", True)
        return action

    def _update_summary(self) -> None:
        statuses = self.session.menu_statuses()
        if "startup" in statuses:
            self.stack.setCurrentIndex(0)
            self.heading.setText("No current dataset")
            return
        self.stack.setCurrentIndex(1)
        title, file_line, rows = _summary_for_session(self.session)
        self.heading.setText(title)
        self.file_label.setText(file_line)
        for index, (name_label, value_label) in enumerate(zip(self.name_labels, self.value_labels)):
            if index < len(rows):
                name, value = rows[index]
                name_label.setText(name)
                value_label.setText(value)
                name_label.show()
                value_label.show()
            else:
                name_label.clear()
                value_label.clear()
                name_label.hide()
                value_label.hide()


def build_main_window(
    session: EEGPrepSession | None = None,
    *,
    all_menus: bool | None = None,
    include_plugins: bool = True,
    native_menu_bar: bool | None = None,
) -> EEGPrepMainWindow:
    """Build an EEGPrep main window without entering the Qt event loop."""
    return EEGPrepMainWindow(
        session=session,
        all_menus=all_menus,
        include_plugins=include_plugins,
        native_menu_bar=native_menu_bar,
    )


def _summary_for_session(session: EEGPrepSession) -> tuple[str, str, list[tuple[str, str]]]:
    eeg = session.current_eeg()
    if session.CURRENTSTUDY == 1 and session.STUDY:
        study = session.STUDY
        return (
            f"STUDY set: {study.get('name', '')}",
            _short_file_line("Study filename", study.get("filepath", ""), study.get("filename", "")),
            [
                ("Study task name", str(study.get("task", ""))),
                ("Nb of subjects", str(max(1, len(study.get("subject", []) or [])))),
                ("Nb of conditions", _per_subject_count(study.get("condition", []) or [])),
                ("Nb of sessions", _per_subject_count(study.get("session", []) or [])),
                ("Nb of groups", _per_subject_count(study.get("group", []) or [])),
                ("Epoch consistency", _epoch_consistency(session.ALLEEG)),
                ("Channels per frame", _unique_values(session.ALLEEG, "nbchan")),
                ("Channel locations", _study_channel_locations(session.ALLEEG)),
                ("Clusters", str(len(study.get("cluster", []) or []))),
                ("Status", _study_status(session.ALLEEG)),
                ("Total size (Mb)", _size_mb(session.ALLEEG)),
            ],
        )
    if isinstance(eeg, list) and len(eeg) > 1:
        indices = ",".join(str(index) for index in session.CURRENTSET)
        return (
            f"Datasets {indices}",
            "Groupname: -(soon)-",
            [
                ("Number of datasets", str(len(eeg))),
                ("Dataset type", _dataset_type(eeg)),
                ("Epoch consistency", _epoch_consistency(eeg)),
                ("Channels per frame", _unique_values(eeg, "nbchan")),
                ("Channel consistency", _channel_consistency(eeg)),
                ("Channel locations", _study_channel_locations(eeg)),
                ("Events (total)", str(sum(_collection_len(item.get("event")) for item in eeg))),
                ("Sampling rate (Hz)", _unique_values(eeg, "srate")),
                ("ICA weights", _yes_no(all(not _empty_array(item.get("icaweights")) for item in eeg))),
                ("Identical ICA", _identical_ica(eeg)),
                ("Total size (Mb)", _size_mb(eeg)),
            ],
        )
    if isinstance(eeg, list):
        eeg = eeg[0] if eeg else {}
    setname = str(eeg.get("setname") or "(no dataset name)")
    prefix = f"#{session.CURRENTSET[0]}: " if session.CURRENTSET else ""
    rows = [
        ("Channels per frame", str(int(eeg.get("nbchan", 0) or 0))),
        ("Frames per epoch", str(int(eeg.get("pnts", 0) or 0))),
        ("Epochs", str(int(eeg.get("trials", 0) or 0))),
        ("Events", "none" if _collection_len(eeg.get("event")) == 0 else str(_collection_len(eeg.get("event")))),
        ("Sampling rate (Hz)", str(int(round(float(eeg.get("srate", 0) or 0))))),
        ("Epoch start (sec)", _format_time(eeg.get("xmin", 0))),
        ("Epoch end (sec)", _format_time(eeg.get("xmax", 0))),
        ("Reference", _reference_state(eeg)),
        ("Channel locations", _channel_location_state(eeg)),
        ("ICA weights", _yes_no(not _empty_array(eeg.get("icasphere")))),
        ("Dataset size (Mb)", _size_mb(eeg)),
    ]
    return prefix + _truncate(setname, 31), _short_file_line("Filename", eeg.get("filepath", ""), eeg.get("filename", "")), rows


def _startup_lines() -> list[str]:
    return [
        "Suggested steps to get started",
        '- Create a new or load an existing dataset:',
        '   Use "File > Import data"           (new)',
        '   Or  "File > Load existing dataset" (load)',
        "   (find tutorial data in sample_data folder)",
        "- If newly imported raw dataset",
        '  "Edit > Channel locations" (look up locations)',
        '  "File > Import event info" (for continuous data)',
        '- Filter data: "Tools > Filter data"',
        '- Reject data: "Tools > Reject data by eye"',
        '- Run ICA: "Tools > Run ICA" (can take time)',
        '- Reject by ICA: "Tools > Reject data using ICA"',
        '- Epoch data: "Tools > Extract epochs"',
        '- Plot ERP: "Plot > Channel ERP > In scalp array"',
    ]


def _main_window_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget#main_panel {{
        background: {BACKEEGLABCOLOR};
        color: {GUITEXTCOLOR};
    }}
    QLabel {{
        background: transparent;
        color: {GUITEXTCOLOR};
        font-family: "Courier New", Courier, monospace;
        font-size: 13px;
    }}
    QLabel#win0 {{
        font-weight: bold;
        font-size: 16px;
    }}
    QLabel#startup_title {{
        font-weight: bold;
    }}
    QFrame#eegprep_frame {{
        border: 1px solid #777777;
        background: {BACKEEGLABCOLOR};
    }}
    """


def _configure_eeglab_label(label: Any, qt_widgets: Any) -> None:
    label.setMinimumWidth(0)
    label.setSizePolicy(qt_widgets.QSizePolicy.Ignored, qt_widgets.QSizePolicy.Fixed)


def _apply_action_metadata(action: Any, spec: MenuItemSpec, qt_gui: Any) -> None:
    action.setMenuRole(qt_gui.QAction.MenuRole.NoRole)
    action.setObjectName(spec.tag or spec.action or spec.label)
    action.setProperty("eegprep_label", spec.label)
    action.setProperty("eegprep_tag", spec.tag or "")
    action.setProperty("eegprep_separator", bool(spec.separator))
    action.setProperty("eegprep_origin", spec.origin)
    action.setProperty("eegprep_checked", bool(spec.checked))
    if spec.action:
        action.setData(spec.action)


def _action_inventory(action: Any) -> dict[str, Any]:
    menu = action.menu()
    children = [_action_inventory(child) for child in menu.actions() if not child.isSeparator()] if menu is not None else []
    return {
        "label": action.text(),
        "enabled": action.isEnabled(),
        "separator": bool(action.property("eegprep_separator")),
        "checked": action.isChecked(),
        "tag": str(action.property("eegprep_tag") or ""),
        "children": children,
    }


def _short_file_line(label: str, filepath: Any, filename: Any) -> str:
    path = "/".join(part for part in (str(filepath or ""), str(filename or "")) if part)
    if not path:
        return f"{label}: none"
    return f"{label}: ...{path[-26:]}" if len(path) > 26 else f"{label}: {path}"


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _format_time(value: Any) -> str:
    numeric = float(value or 0)
    return str(int(numeric)) if numeric == round(numeric) else f"{numeric:6.3f}"


def _reference_state(eeg: dict[str, Any]) -> str:
    chanlocs = _as_list(eeg.get("chanlocs"))
    refs = [str(chan.get("ref")) for chan in chanlocs if isinstance(chan, dict) and chan.get("ref")]
    return refs[0] if refs else str(eeg.get("ref") or "unknown")


def _channel_location_state(eeg: dict[str, Any]) -> str:
    chanlocs = _as_list(eeg.get("chanlocs"))
    if not chanlocs:
        return "No"
    if all(isinstance(chan, dict) and chan.get("theta") in (None, "") for chan in chanlocs):
        return "No (labels only)"
    return "Yes"


def _dataset_type(eeg_list: list[dict[str, Any]]) -> str:
    trials = {int(eeg.get("trials", 1) or 1) for eeg in eeg_list}
    if trials == {1}:
        return "continuous"
    if 1 in trials:
        return "epoched and continuous"
    return "epoched"


def _unique_values(eeg_list: list[dict[str, Any]], key: str) -> str:
    return ", ".join(_format_scalar(value) for value in sorted({eeg.get(key, "") for eeg in eeg_list}))


def _format_scalar(value: Any) -> str:
    if isinstance(value, (int, float)) and float(value) == round(float(value)):
        return str(int(value))
    return str(value)


def _per_subject_count(values: Any) -> str:
    return f"{max(1, len(values))} per subject"


def _epoch_consistency(eeg_list: list[dict[str, Any]]) -> str:
    trials = {int(eeg.get("trials", 1) or 1) for eeg in eeg_list}
    if trials == {1}:
        return "no"
    return "yes" if len(trials) == 1 else "no"


def _channel_consistency(eeg_list: list[dict[str, Any]]) -> str:
    return "yes" if len({int(eeg.get("nbchan", 0) or 0) for eeg in eeg_list}) == 1 else "no"


def _study_channel_locations(eeg_list: list[dict[str, Any]]) -> str:
    states = {_channel_location_state(eeg).lower().startswith("yes") for eeg in eeg_list}
    if states == {True}:
        return "yes"
    if states == {False}:
        return "no"
    return "mixed, yes and no"


def _identical_ica(eeg_list: list[dict[str, Any]]) -> str:
    if not eeg_list:
        return "no"
    first_weights = eeg_list[0].get("icaweights")
    if _empty_array(first_weights):
        return "no"
    for eeg in eeg_list[1:]:
        weights = eeg.get("icaweights")
        if _empty_array(weights):
            return "no"
        if isinstance(first_weights, np.ndarray) and isinstance(weights, np.ndarray):
            if not np.array_equal(first_weights, weights):
                return "no"
        elif first_weights != weights:
            return "no"
    return "yes"


def _study_status(eeg_list: list[dict[str, Any]]) -> str:
    has_ica = all(not _empty_array(eeg.get("icaweights")) for eeg in eeg_list)
    return "Ready to precluster" if has_ica else "Missing ICA dec."


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _empty_array(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, list):
        return len(value) == 0
    return False


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _collection_len(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.size)
    try:
        return len(value)
    except TypeError:
        return 1


def _size_mb(value: Any) -> str:
    if isinstance(value, list):
        return str(round(sum(float(_size_mb(item)) for item in value), 1))
    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, np.ndarray):
            return str(round(data.nbytes / 1e6, 1))
    return "0"

"""Optional PySide6 renderer for EEGLAB-like dialog specs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .spec import CallbackSpec, ControlSpec, DialogSpec


class QtDialogRenderer:
    """Render :class:`DialogSpec` using PySide6 widgets."""

    def run(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        app, dialog, widgets = self.build_dialog(spec, initial_values=initial_values)
        app.processEvents()
        if dialog.exec() != self._QDialog().Accepted:
            return None
        return {tag: self._read_widget(widget) for tag, widget in widgets.items()}

    @staticmethod
    def _QDialog() -> Any:
        try:
            from PySide6.QtWidgets import QDialog
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "PySide6 is required for EEGPrep GUI dialogs. Install it with "
                "`pip install -e .[gui]` or `pip install eegprep[gui]`."
            ) from exc
        return QDialog

    @staticmethod
    def _row_width(geometry: tuple[Any, ...], row: int) -> int:
        value = geometry[min(row, len(geometry) - 1)]
        if isinstance(value, (list, tuple)):
            return max(1, len(value))
        return max(1, int(value))

    def build_dialog(
        self,
        spec: DialogSpec,
        initial_values: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Build a dialog without executing it, for screenshot capture."""
        try:
            from PySide6 import QtCore, QtWidgets
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "PySide6 is required for EEGPrep GUI dialogs. Install it with "
                "`pip install -e .[gui]` or `pip install eegprep[gui]`."
            ) from exc

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        dialog = QtWidgets.QDialog()
        dialog.setObjectName(spec.function_name)
        dialog.setWindowTitle(spec.title)
        self._apply_eeglab_style(dialog)
        layout = QtWidgets.QGridLayout(dialog)
        if spec.function_name == "pop_interp":
            layout.setContentsMargins(23, 14, 25, 13)
        else:
            layout.setContentsMargins(42, 17, 42, 13)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)

        initial_values = initial_values or {}
        widgets: dict[str, Any] = {}
        index = 0
        for row, row_geometry in enumerate(spec.geometry):
            row_width = self._row_width(spec.geometry, row)
            for column in range(row_width):
                if index >= len(spec.controls):
                    break
                control = spec.controls[index]
                widget = self._build_widget(QtWidgets, control, initial_values)
                if control.tag:
                    widgets[control.tag] = widget
                column_span = 1
                if isinstance(row_geometry, (int, float)) and row_geometry == 1 and row_width == 1:
                    column_span = self._max_columns(spec.geometry)
                elif isinstance(row_geometry, (list, tuple)) and len(row_geometry) == 1:
                    column_span = self._max_columns(spec.geometry)
                self._add_widget(QtCore, layout, widget, control, row, column, column_span)
                index += 1

        self._apply_column_stretches(layout, spec)
        for control in spec.controls:
            self._connect_callback(control.callback, widgets)

        self._add_buttons(QtWidgets, layout, spec, dialog)
        dialog.adjustSize()
        self._apply_spec_size(dialog, spec)
        return app, dialog, widgets

    @staticmethod
    def _apply_eeglab_style(dialog: Any) -> None:
        dialog.setStyleSheet(
            """
            QDialog {
                background: #a8c2ff;
                color: #000066;
                font-size: 16px;
            }
            QLabel, QCheckBox, QPushButton, QLineEdit, QComboBox {
                font-size: 16px;
            }
            QLabel, QCheckBox {
                color: #000066;
                background: transparent;
            }
            QLineEdit {
                background: white;
                border: 1px solid #7f7f7f;
                min-width: 217px;
                max-width: 217px;
                min-height: 18px;
                max-height: 18px;
                margin-left: 1px;
                padding: 0 3px;
            }
            QComboBox {
                background: white;
                border: 1px solid #7f7f7f;
                min-width: 217px;
                max-width: 217px;
                min-height: 20px;
                max-height: 20px;
                color: #000066;
            }
            QPushButton {
                background: #eeeeee;
                border: 1px solid #7f7f7f;
                min-width: 79px;
                max-width: 79px;
                min-height: 18px;
                max-height: 18px;
                padding: 0 10px;
                color: #000066;
            }
            QPushButton:disabled {
                color: #b0b0b0;
            }
            QPushButton#events_button {
                min-width: 130px;
                max-width: 130px;
            }
            QPushButton#refbr, QPushButton#exclude_button, QPushButton#refloc_button {
                min-width: 33px;
                max-width: 33px;
                padding: 0;
            }
            QPushButton#interp_nondatchan,
            QPushButton#interp_removedchans,
            QPushButton#interp_datchan,
            QPushButton#interp_selectchan,
            QPushButton#interp_uselist {
                min-width: 434px;
                max-width: 434px;
                padding: 0;
            }
            QDialog#pop_interp QPushButton {
                padding: 0;
            }
            QDialog#pop_interp QLineEdit,
            QDialog#pop_interp QComboBox {
                min-width: 198px;
                max-width: 198px;
            }
            QDialog#pop_reref QLineEdit {
                min-width: 163px;
                max-width: 163px;
            }
            QCheckBox {
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 13px;
                height: 13px;
            }
            QCheckBox::indicator:unchecked {
                background: white;
                border: 1px solid #7f7f7f;
            }
            """
        )

    @staticmethod
    def _apply_column_stretches(layout: Any, spec: DialogSpec) -> None:
        row = QtDialogRenderer._widest_geometry_row(spec.geometry)
        if not isinstance(row, (list, tuple)):
            return
        for column, stretch in enumerate(row):
            layout.setColumnStretch(column, max(1, int(float(stretch) * 100)))

    @staticmethod
    def _add_buttons(QtWidgets: Any, layout: Any, spec: DialogSpec, dialog: Any) -> None:
        button_container = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 18, 0, 0)
        button_layout.setSpacing(16)
        if spec.help_text:
            help_button = QtWidgets.QPushButton("Help")
            help_button.setObjectName("help")
            help_button.clicked.connect(lambda: QtDialogRenderer._show_help(QtWidgets, dialog, spec))
            button_layout.addWidget(help_button)
        button_layout.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel")
        ok_button = QtWidgets.QPushButton("OK")
        cancel_button.setObjectName("cancel")
        ok_button.setObjectName("ok")
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(ok_button)
        layout.addWidget(
            button_container,
            len(spec.geometry),
            0,
            1,
            QtDialogRenderer._max_columns(spec.geometry),
        )

    @staticmethod
    def _max_columns(geometry: tuple[Any, ...]) -> int:
        widths = [
            len(value) if isinstance(value, (list, tuple)) else 1
            for value in geometry
        ]
        return max(widths, default=1)

    @staticmethod
    def _widest_geometry_row(geometry: tuple[Any, ...]) -> Any:
        rows = [value for value in geometry if isinstance(value, (list, tuple))]
        if not rows:
            return ()
        return max(rows, key=len)

    @staticmethod
    def _add_widget(
        QtCore: Any,
        layout: Any,
        widget: Any,
        control: ControlSpec,
        row: int,
        column: int,
        column_span: int,
    ) -> None:
        if control.style.lower() == "edit":
            layout.addWidget(widget, row, column, 1, column_span, QtCore.Qt.AlignLeft)
            return
        layout.addWidget(widget, row, column, 1, column_span)

    @staticmethod
    def _apply_spec_size(dialog: Any, spec: DialogSpec) -> None:
        if spec.size is None:
            return
        dialog.resize(*spec.size)

    def _build_widget(self, QtWidgets: Any, control: ControlSpec, initial_values: Mapping[str, Any]) -> Any:
        style = control.style.lower()
        value = initial_values.get(control.tag, control.value)
        if style == "text":
            widget = QtWidgets.QLabel(control.string)
        elif style == "edit":
            widget = QtWidgets.QLineEdit("" if value is None else str(value))
        elif style == "pushbutton":
            widget = QtWidgets.QPushButton(control.string)
        elif style == "checkbox":
            widget = QtWidgets.QCheckBox(control.string)
            widget.setChecked(bool(value))
        elif style == "popupmenu":
            widget = QtWidgets.QComboBox()
            widget.addItems([item.strip() for item in control.string.split("|")])
            try:
                index = int(value) - 1
            except (TypeError, ValueError):
                index = 0
            if 0 <= index < widget.count():
                widget.setCurrentIndex(index)
        elif style == "spacer":
            widget = QtWidgets.QWidget()
        else:
            raise ValueError(f"Unsupported GUI control style: {control.style}")

        if control.tag:
            widget.setObjectName(control.tag)
        if control.tooltip:
            widget.setToolTip(control.tooltip)
        widget.setEnabled(control.enabled)
        return widget

    def _connect_callback(self, callback: CallbackSpec | None, widgets: dict[str, Any]) -> None:
        if callback is None:
            return
        params = callback.params
        if callback.name == "sync_time_to_samples":
            source = widgets[params["source"]]
            target = widgets[params["target"]]
            multiplier = float(params["srate"])
            source.editingFinished.connect(lambda: self._sync_numeric(source, target, multiplier))
        elif callback.name == "sync_samples_to_time":
            source = widgets[params["source"]]
            target = widgets[params["target"]]
            multiplier = 1.0 / float(params["srate"])
            source.editingFinished.connect(lambda: self._sync_numeric(source, target, multiplier))
        elif callback.name == "select_event_types":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_event_types(button, target, params))
        elif callback.name == "select_channels":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_channels(button, target, params))
        elif callback.name == "set_reref_mode":
            source = widgets[params["source"]]
            source.toggled.connect(lambda checked: self._set_reref_mode(widgets, params["mode"], checked))
        elif callback.name == "select_interp_channels":
            button = widgets[params["button"]]
            target = widgets[params["target"]]
            button.clicked.connect(lambda: self._select_interp_channels(button, target, params))

    @staticmethod
    def _sync_numeric(source: Any, target: Any, multiplier: float) -> None:
        text = source.text().strip()
        if not text:
            target.setText("")
            return
        try:
            value = float(text) * multiplier
        except ValueError:
            return
        target.setText(f"{value:g}")

    @staticmethod
    def _select_event_types(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        from PySide6 import QtWidgets

        event_types = [str(value) for value in params.get("event_types", ())]
        if not event_types:
            return
        current = target.text().strip()
        value, accepted = QtWidgets.QInputDialog.getItem(
            button,
            "Select event type",
            "Event type",
            event_types,
            0,
            editable=False,
        )
        if accepted and value:
            target.setText((current + " " + value).strip())

    @staticmethod
    def _select_channels(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        from eegprep.functions.popfunc.pop_chansel import pop_chansel

        channels = [str(value) for value in params.get("channels", ())]
        if channels:
            _chanlist, value, _allchanstr = pop_chansel(
                channels,
                withindex="on",
                select=target.text().strip(),
                selectionmode=str(params.get("selectionmode", "multiple")),
                parent=button,
            )
            accepted = bool(value)
        else:
            from PySide6 import QtWidgets

            no_channels_message = str(params.get("no_channels_message", "")).strip()
            if no_channels_message:
                QtWidgets.QMessageBox.warning(button, "Warning", no_channels_message)
                return
            value, accepted = QtWidgets.QInputDialog.getText(
                button,
                "Select channel",
                "Channel index or label",
            )
        if not accepted or not value:
            return
        target.setText(value.strip())

    @staticmethod
    def _select_interp_channels(button: Any, target: Any, params: Mapping[str, Any]) -> None:
        from PySide6 import QtWidgets

        from eegprep.functions.popfunc.pop_chansel import pop_chansel

        source = str(params.get("source", "")).lower()
        chanlocs = [dict(chan) for chan in params.get("chanlocs", ())]
        removedchans = [dict(chan) for chan in params.get("removedchans", ())]
        alleeg = [dict(eeg) for eeg in params.get("alleeg", ())]

        if source in {"removedchans", "nondatchan"}:
            labels = [str(chan.get("labels", "")) for chan in removedchans]
            chanlist, chanliststr, _allchanstr = pop_chansel(labels, parent=button)
            if not chanlist:
                return
            selected = [removedchans[index - 1] for index in chanlist]
            chanstr = "EEG.chaninfo.removedchans([" + " ".join(str(index) for index in chanlist) + "])"
            QtDialogRenderer._store_interp_selection(target, selected, chanstr, chanliststr)
            return

        if source == "datchan":
            labels = [str(chan.get("labels", "")) for chan in chanlocs]
            chanlist, chanliststr, _allchanstr = pop_chansel(labels, parent=button)
            if not chanlist:
                return
            selected = [index - 1 for index in chanlist]
            chanstr = "[" + " ".join(str(index - 1) for index in chanlist) + "]"
            QtDialogRenderer._store_interp_selection(target, selected, chanstr, chanliststr)
            return

        dataset_index, accepted = QtWidgets.QInputDialog.getInt(
            button,
            "Choose dataset",
            "Dataset index",
            1,
            1,
            max(1, len(alleeg)),
        )
        if not accepted:
            return
        if dataset_index < 1 or dataset_index > len(alleeg):
            QtWidgets.QMessageBox.warning(button, "Warning", "Wrong index")
            return

        other = alleeg[dataset_index - 1]
        other_chanlocs = [dict(chan) for chan in other.get("chanlocs", ())]
        if source == "selectchan":
            labels = [str(chan.get("labels", "")) for chan in other_chanlocs]
            chanlist, _chanliststr, _allchanstr = pop_chansel(labels, parent=button)
        else:
            chanlist = list(range(1, len(other_chanlocs) + 1))
        if not chanlist:
            return

        current_labels = {str(chan.get("labels", "")).lower() for chan in chanlocs}
        selected_indices = [
            index for index in chanlist
            if str(other_chanlocs[index - 1].get("labels", "")).lower() not in current_labels
        ]
        if not selected_indices:
            QtWidgets.QMessageBox.warning(button, "Warning", "No new channels selected")
            return

        if len(chanlist) == len(other_chanlocs):
            selected = other_chanlocs
            chanstr = f"ALLEEG({dataset_index}).chanlocs"
        else:
            selected_indices = sorted(selected_indices)
            selected = [other_chanlocs[index - 1] for index in selected_indices]
            chanstr = f"ALLEEG({dataset_index}).chanlocs([" + " ".join(str(index) for index in selected_indices) + "])"
        display = " ".join(str(other_chanlocs[index - 1].get("labels", "")) for index in selected_indices)
        QtDialogRenderer._store_interp_selection(target, selected, chanstr, display)

    @staticmethod
    def _store_interp_selection(target: Any, chans: Any, chanstr: str, display: str) -> None:
        target._eegprep_value = {"chans": chans, "chanstr": chanstr}
        target.setText(display.strip())

    @staticmethod
    def _set_reref_mode(widgets: dict[str, Any], mode: str, checked: bool) -> None:
        if not checked:
            if mode == "channels" and "ave" in widgets:
                widgets["ave"].setChecked(True)
            return

        average_mode = mode in {"average", "huber"}
        for tag in ("ave", "huberef", "rerefstr"):
            if tag in widgets:
                widgets[tag].blockSignals(True)
                widgets[tag].setChecked(
                    (tag == "ave" and mode == "average")
                    or (tag == "huberef" and mode == "huber")
                    or (tag == "rerefstr" and mode == "channels")
                )
                widgets[tag].blockSignals(False)

        for tag in ("reref", "refbr", "keepref"):
            if tag in widgets:
                widgets[tag].setEnabled(not average_mode)
        if average_mode and "keepref" in widgets:
            widgets["keepref"].setChecked(False)

    @staticmethod
    def _show_help(QtWidgets: Any, dialog: Any, spec: DialogSpec) -> None:
        from eegprep.functions.guifunc.pophelp import pophelp

        dialog._eegprep_help_dialog = pophelp(spec.help_text or spec.function_name, parent=dialog)

    @staticmethod
    def _read_widget(widget: Any) -> Any:
        if hasattr(widget, "_eegprep_value"):
            return widget._eegprep_value
        if hasattr(widget, "isChecked"):
            return widget.isChecked()
        if hasattr(widget, "currentIndex"):
            return widget.currentIndex() + 1
        if hasattr(widget, "text"):
            return widget.text()
        return None
